package io.improbable.keanu.distributions.discrete;

import java.util.Map;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Categorical<T> implements Distribution<T> {

    private final Map<T, DoubleTensor> selectableValues;

    public static <T> Categorical<T> withParameters(Map<T, DoubleTensor> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<T, DoubleTensor> selectableValues) {
        this.selectableValues = selectableValues;
    }

    public T sample(int[] shape, KeanuRandom random) {
        double sumOfProbabilities = getSumOfProbabilities();
        double p = random.nextDouble();
        double sum = 0;

        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        T value = null;
        for (Map.Entry<T, DoubleTensor> entry : selectableValues.entrySet()) {
            sum += entry.getValue().scalar() / sumOfProbabilities;
            if (p < sum) {
                value = entry.getKey();
                break;
            }
        }
        if (value == null) {
            T[] values = (T[]) selectableValues.keySet().toArray();
            value = values[values.length - 1];
        }

        return value;
    }

    public DoubleTensor logProb(T x) {
        double sumOfProbabilities = getSumOfProbabilities();
        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }
        final double probability = selectableValues.get(x).scalar() / sumOfProbabilities;
        return DoubleTensor.scalar(Math.log(probability));
    }

    private double getSumOfProbabilities() {
        double sumP = 0.0;
        for (DoubleTensor p : selectableValues.values()) {
            sumP += p.scalar();
        }
        return sumP;
    }
}
