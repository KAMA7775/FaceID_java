package org.example;

import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class FaceRecognition {
   private static final Logger logger = LoggerFactory.getLogger(FaceRecognition.class);//экземпляр логгера
   public static void main(String[] args) throws IOException, ModelException, TranslateException {
       logger.info("Starting");//логгирует сообщение с уровнем info (информирует что запускается)
       Path modelDir = Paths.get("C:/Users/Admin/Desktop/facenet");
       Criteria<Image,float[]> criteria = Criteria.builder()//обьект критерия содержит параметры для загрузки модели
               .setTypes(Image.class, float[].class)//тип входных и выходных данных
               .optModelPath(modelDir)//путь к модели
               .optProgress(new ProgressBar())//отслеживание процесса загрузки
               .build();

       logger.info("Loading model from: {}", modelDir.toString());
       try (ZooModel<Image, float[]> model = ModelZoo.loadModel(criteria)) {
           logger.info("Model loaded successfully");

           Image img = ImageFactory.getInstance().fromFile(Paths.get("resources/image/photo_2024-05-05_11-29-32.jpg"));
           logger.info("Image loaded: {}", "resources/image/photo_2024-05-05_11-29-32.jpg");

           Translator<Image, float[]> translator = new Translator<Image, float[]>() {
               @Override
               public float[] processOutput(TranslatorContext translatorContext, NDList ndList) {
                   return ndList.singletonOrThrow().toFloatArray();
               }

               @Override
               public NDList processInput(TranslatorContext translatorContext, Image image) {
                   NDArray array = image.toNDArray(translatorContext.getNDManager());
                   array = new Resize(160, 160).transform(array);
                   array = new ToTensor().transform(array);
                   return new NDList(array);
               }

               @Override
               public Batchifier getBatchifier() {
                   return null;
               }
           };
           float[] embedding = model.newPredictor(translator).predict(img);
           logger.info("Embedding generated: {}", Arrays.toString(embedding));
       }
       catch(ModelNotFoundException | TranslateException e){
           logger.error("Error",e);
       }
           logger.info("Face Recognition finished");


   }
}