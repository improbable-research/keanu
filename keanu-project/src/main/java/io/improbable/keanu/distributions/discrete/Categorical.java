package io.improbable.keanu.distributions.discrete;

import java.util.Map;

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

    public Tensor<T> sample(int[] shape, KeanuRandom random) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(shape);
        if (!sumOfProbabilities.lessThanOrEqual(0.).allFalse()) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        double p = random.nextDouble();
        DoubleTensor sum = DoubleTensor.zeros(shape);
        Tensor<T> sample = Tensor.placeHolder(shape);
        BooleanTensor valueSetSoFar = BooleanTensor.falses(shape);

        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            sum.plusInPlace(entry.getValue().div(sumOfProbabilities));

            BooleanTensor mask = valueSetSoFar.not().andInPlace(sum.greaterThan(p));

            sample = mask.setIf(Tensor.scalar(entry.getKey()), sample);
            valueSetSoFar.orInPlace(mask);

            if (valueSetSoFar.allTrue()) {
                break;
            }
        }

        if (!valueSetSoFar.allTrue()) {
            T[] values = (T[]) selectableValues.keySet().toArray();
            sample = Tensor.create(values[values.length - 1], shape);
        }

        return sample;
    }

    public DoubleTensor logProb(Tensor<T> x) {
        DoubleTensor sumOfProbabilities = getSumOfProbabilities(x.getShape());
        if (!sumOfProbabilities.lessThanOrEqual(0.).allFalse()) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        DoubleTensor logProb = DoubleTensor.zeros(x.getShape());
        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            logProb.plusInPlace(x.elementwiseEquals(Tensor.create(entry.getKey(), x.getShape())).toDoubleMask().timesInPlace(entry.getValue().div(sumOfProbabilities).logInPlace()));
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
