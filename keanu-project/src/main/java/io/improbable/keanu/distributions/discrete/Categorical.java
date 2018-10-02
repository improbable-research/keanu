package io.improbable.keanu.distributions.discrete;

import java.util.Map;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Categorical<T> implements Distribution<GenericTensor<T>> {

    private final Map<T, DoubleTensor> selectableValues;

    public static <T> Categorical<T> withParameters(Map<T, DoubleTensor> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<T, DoubleTensor> selectableValues) {
        this.selectableValues = selectableValues;
    }

    public GenericTensor<T> sample(int[] shape, KeanuRandom random) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(shape);
        if (!sumOfProbabilities.lessThanOrEqual(0.).allFalse()) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        double p = random.nextDouble();
        DoubleTensor sum = DoubleTensor.zeros(shape);
        GenericTensor<T> value = new GenericTensor<>(shape);

        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            sum.plusInPlace(entry.getValue().div(sumOfProbabilities));

            BooleanTensor mask = sum.greaterThan(p);
            value.setWithMaskInPlace(mask.toDoubleMask(), entry.getKey());

            if (mask.allTrue()) {
                break;
            }
        }

        if (value.isNull()) {
            value = new GenericTensor<>((T[]) selectableValues.keySet().toArray(), shape);
        }

        return value;
    }

    public DoubleTensor logProb(GenericTensor<T> x) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(x.getShape());
        if (!sumOfProbabilities.lessThanOrEqual(0.).allFalse()) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        DoubleTensor logProb = DoubleTensor.zeros(x.getShape());
        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            logProb.plusInPlace(x.equalsMask(entry.getKey()).timesInPlace(entry.getValue().div(sumOfProbabilities).logInPlace()));
        }
        return logProb;
    }

    private DoubleTensor getSumOfProbabilities(int[] shape) {
        DoubleTensor sumP = DoubleTensor.zeros(shape);
        for (DoubleTensor p : selectableValues.values()) {
            sumP.plusInPlace(p);
        }
        return sumP;
    }
}
