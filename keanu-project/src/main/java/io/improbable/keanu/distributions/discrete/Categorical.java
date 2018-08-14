package io.improbable.keanu.distributions.discrete;

import java.util.Map;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Categorical<T> {

    private final Map<T, DoubleVertex> selectableValues;

    /**
     * <h3>Categorical (Generalised Bernoulli Distribution) Distribution</h3>
     *
     * @param selectableValues a mapping of category T to event probability
     * @see <a href="https://en.wikipedia.org/wiki/Categorical_distribution">Wikipedia</a>
     */
    public static <T> Categorical withParameters(Map<T, DoubleVertex> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<T, DoubleVertex> selectableValues) {
        this.selectableValues = selectableValues;
    }

    public T sample(KeanuRandom random) {
        double sumOfProbabilities = getSumOfProbabilities();
        double p = random.nextDouble();
        double sum = 0;

        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        T value = null;
        for (Map.Entry<T, DoubleVertex> entry : selectableValues.entrySet()) {
            sum += entry.getValue().getValue().scalar() / sumOfProbabilities;
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

    public double logProb(T x) {
        double sumOfProbabilities = getSumOfProbabilities();
        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }
        final double probability = selectableValues.get(x).getValue().scalar() / sumOfProbabilities;
        return Math.log(probability);
    }

    private double getSumOfProbabilities() {
        double sumP = 0.0;
        for (DoubleVertex p : selectableValues.values()) {
            sumP += p.getValue().scalar();
        }
        return sumP;
    }

}