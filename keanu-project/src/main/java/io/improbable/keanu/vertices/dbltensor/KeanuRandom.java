package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.List;

public class KeanuRandom {

    static {
        System.setProperty("dtype", "double");
    }

    private final Random nd4jRandom;

    public KeanuRandom() {
        nd4jRandom = new DefaultRandom();
    }

    public KeanuRandom(long seed) {
        nd4jRandom = new DefaultRandom(seed);
    }

    public DoubleTensor nextDouble(int[] shape) {
        return new Nd4jDoubleTensor(nd4jRandom.nextDouble(shape));
    }

    public DoubleTensor nextGaussian(int[] shape) {
        return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
    }

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {
        List<Double> samples = exploreIndexesAndSampleGamma(shape, a, theta, k);
        return createTensorFromList(samples, shape);
    }

    private List<Double> exploreIndexesAndSampleGamma(int[] shape, DoubleTensor... hyperParamaters) {
        List<Double> samples = new ArrayList<>();
        int[] results = new int[shape.length];
        iterateThroughShape(0, shape.length, shape, results, samples, hyperParamaters);
        return samples;
    }

    private void iterateThroughShape(int count, int length, int[] size, int[] result, List<Double> samples, DoubleTensor... hyperParamaters) {
        if (count >= length) {
            Double sample = KeanuRandomSampling.gammaSample(
                hyperParamaters[0].getValue(result),
                hyperParamaters[1].getValue(result),
                hyperParamaters[2].getValue(result),
                nd4jRandom
            );
            samples.add(sample);
            return;
        }
        for (int i = 0; i < size[count]; i++) {
            result[count] = i;
            iterateThroughShape(count + 1, length, size, result, samples, hyperParamaters);
        }
    }

    private DoubleTensor createTensorFromList(List<Double> list, int[] shape) {
        double[] values = list.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(values, shape);
    }

}
