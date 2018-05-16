package io.improbable.keanu.vertices.dbltensor;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.*;
import static java.lang.Math.pow;

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
        List<Double> gammaValues = new ArrayList<>();
        List<List<Integer>> possibleIndexes = exploreIndexes(shape);

        for (List<Integer> index : possibleIndexes) {
            int[] currentDimension = index.stream().mapToInt(i -> i).toArray();
            double sample = gammaSample(
                a.getValue(currentDimension),
                theta.getValue(currentDimension),
                k.getValue(currentDimension),
                nd4jRandom
            );
            gammaValues.add(sample);
        }

        return createTensorFromList(gammaValues, shape);
    }

    private List<List<Integer>> exploreIndexes(int[] shape) {
        List<List<Integer>> possibleIndexes = new ArrayList<>();
        int[] results = new int[shape.length];
        iterate(0, shape.length, shape, results, possibleIndexes);
        return possibleIndexes;
    }

    private void iterate(int count, int length, int[] size, int[] res, List<List<Integer>> dimensions) {
        if (count >= length) {
            Integer[] result = ArrayUtils.toObject(res);
            dimensions.add(Arrays.asList(result));
            return;
        }
        for (int i = 0; i < size[count]; i++) {
            res[count] = i;
            iterate(count + 1, length, size, res, dimensions);
        }
    }

    private DoubleTensor createTensorFromList(List<Double> list, int[] shape) {
        double[] values = list.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(values, shape);
    }

    public static double gammaSample(double a, double theta, double k, Random random) {
        if (theta <= 0. || k <= 0.) {
            throw new IllegalArgumentException("Invalid value for theta or k. Theta: " + theta + ". k: " + k);
        }
        final double M_E = 0.577215664901532860606512090082;
        final double A = 1. / sqrt(2. * k - 1.);
        final double B = k - log(4.);
        final double Q = k + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + k / M_E;

        if (k < 1.) {
            return sampleWhileKLessThanOne(C, k, a, theta, random);
//        } else if (k == 1.0) return Exponential.sample(a, theta, random);
        }
        else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = k * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return a + theta * y;
            }
        }
    }

    private static double sampleWhileKLessThanOne(double c, double k, double a, double theta, Random random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / k);
                if (random.nextDouble() <= pow(y, k - 1.)) return a + theta * y;
            } else {
                double y = pow(p, 1. / k);
                if (random.nextDouble() <= exp(-y)) return a + theta * y;
            }
        }
    }

}
