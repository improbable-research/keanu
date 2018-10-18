package io.improbable.keanu.distributions.discrete;

import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Categorical<T> implements Distribution<Tensor<T>> {

    private final Map<T, DoubleTensor> selectableValues;

    public static <T> Categorical<T> withParameters(Map<T, DoubleTensor> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<T, DoubleTensor> selectableValues) {
        this.selectableValues = selectableValues;
    }

    public Tensor<T> sample(long[] shape, KeanuRandom random) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(shape);
        if (containsNonPositiveEntry(sumOfProbabilities)) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        double p = random.nextDouble();
        DoubleTensor sum = DoubleTensor.zeros(shape);

        Tensor<T> sample = Tensor.create(null, shape);
        BooleanTensor sampleValuesSetSoFar = BooleanTensor.falses(shape);

        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            sum = sum.plus(entry.getValue().div(sumOfProbabilities));

            BooleanTensor maskForUnassignedSampleValues = sampleValuesSetSoFar.xor(sum.greaterThan(p));
            sample = maskForUnassignedSampleValues.where(Tensor.scalar(entry.getKey()), sample);

            sampleValuesSetSoFar.orInPlace(maskForUnassignedSampleValues);

            if (sampleValuesSetSoFar.allTrue()) {
                break;
            }
        }

        if (!sampleValuesSetSoFar.allTrue()) {
            Set<T> values = selectableValues.keySet();
            T lastValue = ImmutableList.copyOf(values).get(values.size() - 1);
            sample = Tensor.create(lastValue, shape);
        }

        return sample;
    }

    public DoubleTensor logProb(Tensor<T> x) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(x.getShape());
        if (containsNonPositiveEntry(sumOfProbabilities)) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        DoubleTensor logProb = DoubleTensor.zeros(x.getShape());
        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            DoubleTensor xEqualToEntryKeyMask = x.elementwiseEquals(Tensor.create(entry.getKey(), x.getShape())).toDoubleMask();
            logProb = logProb.plus(xEqualToEntryKeyMask.timesInPlace(entry.getValue().div(sumOfProbabilities).logInPlace()));
        }
        return logProb;
    }

    private boolean containsNonPositiveEntry(DoubleTensor sumOfProbabilities) {
        return !sumOfProbabilities.lessThanOrEqual(0.).allFalse();
    }

    private DoubleTensor getSumOfProbabilities(long[] shape) {
        DoubleTensor sumP = DoubleTensor.zeros(shape);
        for (DoubleTensor p : selectableValues.values()) {
            sumP = sumP.plus(p);
        }
        return sumP;
    }
}
