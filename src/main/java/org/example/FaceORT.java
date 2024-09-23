package org.example;

import ai.onnxruntime.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
//onixxruntime library
public class FaceORT {
    private static final Logger logger = Logger.getLogger(FaceORT.class.getName());
        private static final Pattern splitPattern = Pattern.compile("\\s+");
        private static class SparseData {
            public final int[] labels;
            public final List<int[]> indices;
            public final List<float[]> values;

            public SparseData(int[] labels, List<int[]> indices, List<float[]> values) {
                this.labels = labels;
                this.indices = Collections.unmodifiableList(indices);
                this.values = Collections.unmodifiableList(values);
            }
        }
        private static int[] convertInts(List<Integer> list) {
            int[] output = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                output[i] = list.get(i);
            }
            return output;
        }


        private static float[] convertFloats(List<Float> list) {
            float[] output = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                output[i] = list.get(i);
            }
            return output;
        }

        private static SparseData load(String path) throws IOException {
            int pos = 0;
            List<int[]> indices = new ArrayList<>();
            List<float[]> values = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            String line;
            int maxFeatureID = Integer.MIN_VALUE;
            try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
                for (; ; ) {
                    line = reader.readLine();
                    if (line == null) {
                        break;
                    }
                    pos++;
                    String[] fields = splitPattern.split(line);
                    int lastID = -1;
                    try {
                        boolean valid = true;
                        List<Integer> curIndices = new ArrayList<>();
                        List<Float> curValues = new ArrayList<>();
                        for (int i = 1; i < fields.length && valid; i++) {
                            int ind = fields[i].indexOf(':');
                            if (ind < 0) {
                                logger.warning(String.format("Weird line at %d", pos));
                                valid = false;
                            }
                            String ids = fields[i].substring(0, ind);
                            int id = Integer.parseInt(ids);
                            curIndices.add(id);
                            if (maxFeatureID < id) {
                                maxFeatureID = id;
                            }
                            float val = Float.parseFloat(fields[i].substring(ind + 1));
                            curValues.add(val);
                            if (id <= lastID) {
                                logger.warning(String.format("Repeated features at line %d", pos));
                                valid = false;
                            } else {
                                lastID = id;
                            }
                        }
                        if (valid) {
                            // Store the label
                            labels.add(Integer.parseInt(fields[0]));
                            // Store the features
                            indices.add(convertInts(curIndices));
                            values.add(convertFloats(curValues));
                        } else {
                            throw new IOException("Invalid LibSVM format file at line " + pos);
                        }
                    } catch (NumberFormatException ex) {
                        logger.warning(String.format("Weird line at %d", pos));
                        throw new IOException("Invalid LibSVM format file", ex);
                    }
                }
            }

            logger.info(
                    "Loaded "
                            + maxFeatureID
                            + " features, "
                            + labels.size()
                            + " samples, from + '"
                            + path
                            + "'.");
            return new SparseData(convertInts(labels), indices, values);
        }
        public static float[] softmax(float[] input) {
            double[] tmp = new double[input.length];
            double sum = 0.0;
            for (int i = 0; i < input.length; i++) {
                double val = Math.exp(input[i]);
                sum += val;
                tmp[i] = val;
            }

            float[] output = new float[input.length];
            for (int i = 0; i < output.length; i++) {
                output[i] = (float) (tmp[i] / sum);
            }

            return output;
        }
        public static void zeroData(float[][][][] data) {
            // Zero the array
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[i].length; j++) {
                    for (int k = 0; k < data[i][j].length; k++) {
                        Arrays.fill(data[i][j][k], 0.0f);
                    }
                }
            }
        }
        public static void writeData(float[][][][] data, int[] indices, float[] values) {
            zeroData(data);

            for (int m = 0; m < indices.length; m++) {
                int i = (indices[m]) / 28;
                int j = (indices[m]) % 28;
                data[0][0][i][j] = values[m] / 255;
            }

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    data[0][0][i][j] = (data[0][0][i][j] - 0.1307f) / 0.3081f;
                }
            }
        }
        public static void zeroDataSKL(float[][] data) {
            // Zero the array
            for (int i = 0; i < data.length; i++) {
                Arrays.fill(data[i], 0.0f);
            }
        }
        public static void writeDataSKL(float[][] data, int[] indices, float[] values) {
            zeroDataSKL(data);

            for (int m = 0; m < indices.length; m++) {
                data[0][indices[m]] = values[m];
            }
        }
        public static int pred(float[] probabilities) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int idx = 0;
            for (int i = 0; i < probabilities.length; i++) {
                if (probabilities[i] > maxVal) {
                    maxVal = probabilities[i];
                    idx = i;
                }
            }
            return idx;
        }

        public static void main(String[] args) throws OrtException, IOException {
            if (args.length < 2 || args.length > 3) {
                System.out.println("Usage: ScoreMNIST <model-path> <test-data> <optional:scikit-learn-flag>");
                System.out.println("The test data input should be a libsvm format version of MNIST.");
                return;
            }

            OrtEnvironment env = OrtEnvironment.getEnvironment();
            try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {

                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

                logger.info("Loading model from " + args[0]);
                try (OrtSession session = env.createSession(args[0], opts)) {

                    logger.info("Inputs:");
                    for (NodeInfo i : session.getInputInfo().values()) {
                        logger.info(i.toString());
                    }

                    logger.info("Outputs:");
                    for (NodeInfo i : session.getOutputInfo().values()) {
                        logger.info(i.toString());
                    }

                    SparseData data = load(args[1]);

                    float[][][][] testData = new float[1][1][28][28];
                    float[][] testDataSKL = new float[1][780];

                    int correctCount = 0;
                    int[][] confusionMatrix = new int[10][10];

                    String inputName = session.getInputNames().iterator().next();

                    for (int i = 0; i < data.labels.length; i++) {
                        if (args.length == 3) {
                            writeDataSKL(testDataSKL, data.indices.get(i), data.values.get(i));
                        } else {
                            writeData(testData, data.indices.get(i), data.values.get(i));
                        }

                        try (OnnxTensor test =
                                     OnnxTensor.createTensor(env, args.length == 3 ? testDataSKL : testData);
                             OrtSession.Result output = session.run(Collections.singletonMap(inputName, test))) {

                            int predLabel;

                            if (args.length == 3) {
                                long[] labels = (long[]) output.get(0).getValue();
                                predLabel = (int) labels[0];
                            } else {
                                float[][] outputProbs = (float[][]) output.get(0).getValue();
                                predLabel = pred(outputProbs[0]);
                            }
                            if (predLabel == data.labels[i]) {
                                correctCount++;
                            }

                            confusionMatrix[data.labels[i]][predLabel]++;

                            if (i % 2000 == 0) {
                                logger.log(Level.INFO, "Cur accuracy = " + ((float) correctCount) / (i + 1));
                                logger.log(Level.INFO, "Output type = " + output.get(0).toString());
                                if (args.length == 3) {
                                    logger.log(Level.INFO, "Output type = " + output.get(1).toString());
                                    logger.log(Level.INFO, "Output value = " + output.get(1).getValue().toString());
                                }
                            }
                        }
                    }

                    logger.info("Final accuracy = " + ((float) correctCount) / data.labels.length);

                    StringBuilder sb = new StringBuilder();
                    sb.append("Label");
                    for (int i = 0; i < confusionMatrix.length; i++) {
                        sb.append(String.format("%1$5s", "" + i));
                    }
                    sb.append("\n");

                    for (int i = 0; i < confusionMatrix.length; i++) {
                        sb.append(String.format("%1$5s", "" + i));
                        for (int j = 0; j < confusionMatrix[i].length; j++) {
                            sb.append(String.format("%1$5s", "" + confusionMatrix[i][j]));
                        }
                        sb.append("\n");
                    }

                    System.out.println(sb.toString());
                }
            }

            logger.info("Done!");
        }
    }
