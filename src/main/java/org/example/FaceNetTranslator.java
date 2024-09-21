package org.example;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class FaceNetTranslator implements Translator<Image, float[]> {

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow().toFloatArray();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDArray array = input.toNDArray(ctx.getNDManager());
        array = new Resize(160, 160).transform(array);
        array = new ToTensor().transform(array);
        return new NDList(array);
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}

