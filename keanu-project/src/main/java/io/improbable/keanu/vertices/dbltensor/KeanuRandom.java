package io.improbable.keanu.vertices.dbltensor;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

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

    public DoubleTensor nextLaplace(int[] shape, DoubleTensor mu, DoubleTensor beta) {
        List<Double> laplaceValues = new ArrayList<>();
        List<List<Integer>> possibleIndexes = exploreIndexes(shape);

        for (List<Integer> index : possibleIndexes) {
            int[] currentDimension = index.stream().mapToInt(i -> i).toArray();
            double sample = laplaceSample(
                mu.getValue(currentDimension),
                beta.getValue(currentDimension),
                nd4jRandom
            );
            laplaceValues.add(sample);
        }

        return createTensorFromList(laplaceValues, shape);
    }

    private List<List<Integer>> exploreIndexes(int[] shape) {
        List<List<Integer>> possibleIndexes = new ArrayList<>();
        int[] results = new int[shape.length];
        iterateThroughShape(0, shape.length, shape, results, possibleIndexes);
        return possibleIndexes;
    }

    private void iterateThroughShape(int count, int length, int[] size, int[] result, List<List<Integer>> dimensions) {
        if (count >= length) {
            Integer[] res = ArrayUtils.toObject(result);
            dimensions.add(Arrays.asList(res));
            return;
        }
        for (int i = 0; i < size[count]; i++) {
            result[count] = i;
            iterateThroughShape(count + 1, length, size, result, dimensions);
        }
    }

    private DoubleTensor createTensorFromList(List<Double> list, int[] shape) {
        double[] values = list.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(values, shape);
    }

    private static double laplaceSample(double mu, double beta, Random random) {
        if (beta <= 0.0) {
            throw new IllegalArgumentException("Invalid value for beta: " + beta);
        }
        if (random.nextDouble() > 0.5) {
            return mu + beta * Math.log(random.nextDouble());
        } else {
            return mu - beta * Math.log(random.nextDouble());
        }
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
