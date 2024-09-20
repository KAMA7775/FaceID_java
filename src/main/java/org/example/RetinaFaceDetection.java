package org.example;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
//для обнаружения лиц на изображении с использованием модели RetinaFace
public final class RetinaFaceDetection {

    private static final Logger logger = LoggerFactory.getLogger(RetinaFaceDetection.class);

    private RetinaFaceDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = RetinaFaceDetection.predict();
        logger.info("{}", detection);// метод predict()чтобы выполнить обнаружение лиц
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path facePath = Paths.get("resources/image/photo_2024-05-05_11-57-05.jpg");
        Image img = ImageFactory.getInstance().fromFile(facePath);
// параметры для обнаружения лиц
        double confThresh = 0.85f;
        double nmsThresh = 0.45f;
        double[] variance = {0.1f, 0.2f};
        int topK = 5000;
        int[][] scales = {{16, 32}, {64, 128}, {256, 512}};
        int[] steps = {8, 16, 32};
        FaceDetectionTranslator translator =
                new FaceDetectionTranslator(confThresh, nmsThresh, variance, topK, scales, steps);

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelUrls("C:/Users/Admin/Desktop/facenet")
                        // Load model from local file, e.g:
                        .optModelName("retinaface") // specify model file prefix
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch") // Use PyTorch engine
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            DetectedObjects detection = predictor.predict(img);
            saveBoundingBoxImage(img, detection);
            return detection;
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("Screenshort.png");
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Face detection result image has been saved in: {}", imagePath);
    }
}

