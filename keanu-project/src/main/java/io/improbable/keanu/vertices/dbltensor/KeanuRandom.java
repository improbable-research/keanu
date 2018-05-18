package io.improbable.keanu.vertices.dbltensor;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static io.improbable.keanu.vertices.dbltensor.KeanuRandomSampling.gammaSample;

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
        List<Double> samples = exploreIndexesAndSample(shape, a, theta, k);
        return createTensorFromList(samples, shape);
    }

    private List<Double> exploreIndexesAndSample(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {
        List<Double> samples = new ArrayList<>();
        int[] results = new int[shape.length];
        iterateThroughShape(0, shape.length, shape, results, samples, a, theta, k);
        return samples;
    }

    private void iterateThroughShape(int count, int length, int[] size, int[] result, List<Double> samples, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {
        if (count >= length) {
            double sample = gammaSample(
                a.getValue(result),
                theta.getValue(result),
                k.getValue(result),
                nd4jRandom
            );
            samples.add(sample);
            return;
        }
        for (int i = 0; i < size[count]; i++) {
            result[count] = i;
            iterateThroughShape(count + 1, length, size, result, samples, a, theta, k);
        }
    }

    private DoubleTensor createTensorFromList(List<Double> list, int[] shape) {
        double[] values = list.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(values, shape);
    }

}
