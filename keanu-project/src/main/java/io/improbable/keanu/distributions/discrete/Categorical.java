package io.improbable.keanu.distributions.discrete;

import java.util.Map;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Categorical<N, T extends Tensor<N>> implements Distribution<T> {

    private final Map<N, DoubleTensor> selectableValues;

    public static <N, T extends Tensor<N>> Categorical<N, T> withParameters(Map<N, DoubleTensor> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<N, DoubleTensor> selectableValues) {
        this.selectableValues = selectableValues;
    }

    public T sample(long[] shape, KeanuRandom random) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(shape);
        if (containsNonPositiveEntry(sumOfProbabilities)) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        double p = random.nextDouble();
        DoubleTensor sum = DoubleTensor.zeros(shape);

        T sample = (T) Tensor.create(selectableValues.keySet().iterator().next(), shape);
        BooleanTensor sampleSetSoFar = BooleanTensor.falses(shape);

        for (Map.Entry<N, DoubleTensor> entry : selectableValues.entrySet()) {
            sum.plusInPlace(entry.getValue().div(sumOfProbabilities));

            BooleanTensor mask = sampleSetSoFar.not().andInPlace(sum.greaterThan(p));
            sample = (T) mask.setIf(Tensor.scalar(entry.getKey()), sample);

            sampleSetSoFar.orInPlace(mask);

            if (sampleSetSoFar.allTrue()) {
                break;
            }
        }

        if (!sampleSetSoFar.allTrue()) {
            N[] values = (N[]) selectableValues.keySet().toArray();
            sample = (T) Tensor.create(values[values.length - 1], shape);
        }

        return sample;
    }

    public DoubleTensor logProb(T x) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(x.getShape());
        if (containsNonPositiveEntry(sumOfProbabilities)) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        DoubleTensor logProb = DoubleTensor.zeros(x.getShape());
        for (Map.Entry<N, DoubleTensor> entry : selectableValues.entrySet()) {
            DoubleTensor xEqualToEntryKeyMask = x.elementwiseEquals(Tensor.create(entry.getKey(), x.getShape())).toDoubleMask();
            logProb.plusInPlace(xEqualToEntryKeyMask.timesInPlace(entry.getValue().div(sumOfProbabilities).logInPlace()));
        }
        return logProb;
    }

    private boolean containsNonPositiveEntry(DoubleTensor sumOfProbabilities) {
        return !sumOfProbabilities.lessThanOrEqual(0.).allFalse();
    }

    private DoubleTensor getSumOfProbabilities(int[] shape) {
        DoubleTensor sumP = DoubleTensor.zeros(shape);
        for (DoubleTensor p : selectableValues.values()) {
            sumP.plusInPlace(p);
        }
        return sumP;
    }
}
