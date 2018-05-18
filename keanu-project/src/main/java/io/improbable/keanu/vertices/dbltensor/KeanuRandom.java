package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;

public class KeanuRandom {

    private static final AtomicReference<KeanuRandom> DEFAULT_RANDOM = new AtomicReference<>();

    static {
        System.setProperty("dtype", "double");

        String randomSeed = System.getProperty("io.improbable.keanu.defaultRandom.seed");

        if (randomSeed != null) {
            final long seed = Long.parseLong(randomSeed);
            DEFAULT_RANDOM.set(new KeanuRandom(seed));
        } else {
            DEFAULT_RANDOM.set(new KeanuRandom());
        }
    }

    public static KeanuRandom getDefaultRandom() {
        return DEFAULT_RANDOM.get();
    }

    public static void setDefaultRandomSeed(long seed) {
        DEFAULT_RANDOM.set(new KeanuRandom(seed));
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

    public double nextDouble() {
        return nd4jRandom.nextDouble();
    }

    public DoubleTensor nextGaussian(int[] shape) {
        return new Nd4jDoubleTensor(nd4jRandom.nextGaussian(shape));
    }

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k) {
        List<Double> samples = exploreIndexesAndSample(shape, KeanuRandomSampling::gammaSample, a, theta, k);
        return createTensorFromList(samples, shape);
    }

    public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {
        List<Double> samples = exploreIndexesAndSample(shape, KeanuRandomSampling::laplaceSample, mu, beta);
        return createTensorFromList(samples, shape);
    }

    private List<Double> exploreIndexesAndSample(
        int[] shape,
        BiFunction<List<Double>, Random, Double> samplingFunc,
        DoubleTensor... hyperParameters) {

        List<Double> samples = new ArrayList<>();
        int[] results = new int[shape.length];
        iterateThroughShape(0, shape.length, shape, results, samples, samplingFunc, hyperParameters);
        return samples;
    }

    private void iterateThroughShape(
        int count,
        int length,
        int[] size,
        int[] result,
        List<Double> samples,
        BiFunction<List<Double>, Random, Double> samplingFunc,
        DoubleTensor... hyperParameters) {

        List<Double> hyperParamValues = new ArrayList<>();
        for (DoubleTensor hyperParameter : hyperParameters) {
            hyperParamValues.add(hyperParameter.getValue(result));
        }

        if (count >= length) {
            Double sample = samplingFunc.apply(hyperParamValues, nd4jRandom);
            samples.add(sample);
            return;
        }
        for (int i = 0; i < size[count]; i++) {
            result[count] = i;
            iterateThroughShape(count + 1, length, size, result, samples, samplingFunc, hyperParameters);
        }
    }

    private DoubleTensor createTensorFromList(List<Double> list, int[] shape) {
        double[] values = list.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(values, shape);
    }

    public double nextGaussian() {
        return nd4jRandom.nextGaussian();
    }

    public boolean nextBoolean() {
        return nd4jRandom.nextBoolean();
    }

    public int nextInt(int maxExclusive) {
        return nd4jRandom.nextInt(maxExclusive);
    }
}
