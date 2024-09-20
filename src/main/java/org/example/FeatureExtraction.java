package org.example;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
//FeatureExtraction предназначен для извлечения признаков (features) из изображения с использованием PyTorch
public final class FeatureExtraction {

    private static final Logger logger = LoggerFactory.getLogger(FeatureExtraction.class);

    private FeatureExtraction() {
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("C:\\Users\\Admin\\Desktop\\FaceDJL\\src\\main\\resources\\image\\photo_2024-05-05_11-29-28.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        float[] feature = FeatureExtraction.predict(img);
        if (feature != null) {
            logger.info(Arrays.toString(feature));
        }
    }

    public static float[] predict(Image img)
            throws IOException, ModelException, TranslateException {
        img.getWrappedImage();
        Criteria<Image, float[]> criteria =
                Criteria.builder()
                        .setTypes(Image.class, float[].class)
                        .optModelUrls("\"C:\\Users\\Admin\\Desktop\\facenet\"")
                        .optModelName("facenet") // specify model file prefix
                        .optTranslator(new FaceFeatureTranslator())
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch")
                        .build();

        try (ZooModel<Image, float[]> model = criteria.loadModel()) {
            Predictor<Image, float[]> predictor = model.newPredictor();
            return predictor.predict(img);
        }
    }
    private static final class FaceFeatureTranslator implements Translator<Image, float[]> {

        FaceFeatureTranslator() {}

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            Pipeline pipeline = new Pipeline();
            pipeline
                    // .add(new Resize(160))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                    new float[] {
                                            128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                    }));

            return pipeline.transform(new NDList(array));
        }


        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            float[][] embeddings =
                    result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
            float[] feature = new float[embeddings.length];
            for (int i = 0; i < embeddings.length; i++) {
                feature[i] = embeddings[i][0];
            }
            return feature;
        }
    }
}