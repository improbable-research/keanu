package io.improbable.keanu.vertices.dbltensor;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

}
