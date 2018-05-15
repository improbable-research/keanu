package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.distributions.continuous.Gamma;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
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

    public DoubleTensor nextGamma(int[] shape, DoubleTensor a, DoubleTensor theta, DoubleTensor k, java.util.Random random) {
        List<Double> gammaValues = new ArrayList<>();
        List<List<Integer>> allDimensionCombinations = new ArrayList<>();

        int[] results = new int[shape.length];
        iterate(0, shape.length, shape, results, allDimensionCombinations);

        for (List<Integer> dimension : allDimensionCombinations) {
            int[] currentDimension = dimension.stream().mapToInt(i -> i).toArray();
            gammaValues.add(
                Gamma.sample(
                    a.getValue(currentDimension),
                    theta.getValue(currentDimension),
                    k.getValue(currentDimension),
                    random
                )
            );
        }
        double[] gammaSamples = gammaValues.stream().mapToDouble(d -> d).toArray();
        return Nd4jDoubleTensor.create(gammaSamples, shape);
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

}
