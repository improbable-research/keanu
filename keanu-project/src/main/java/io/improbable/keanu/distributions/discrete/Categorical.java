package io.improbable.keanu.distributions.discrete;

import java.util.Map;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see <a href="https://en.wikipedia.org/wiki/Categorical_distribution">Wikipedia</a>
 */
public class Categorical<T> {

    private final Map<T, DoubleVertex> selectableValues;

    /**
     * @param selectableValues a mapping of category T to event probability, the total sum of probabilities must not be equal to 0
     * @param <T>              Category object type
     * @return an instance of {@link Categorical}
     */
    public static <T> Categorical<T> withParameters(Map<T, DoubleVertex> selectableValues) {
        return new Categorical<>(selectableValues);
    }

    private Categorical(Map<T, DoubleVertex> selectableValues) {
        this.selectableValues = selectableValues;
    }

    /**
     * @param random {@link KeanuRandom}
     * @return a sample of T
     * @throws IllegalArgumentException if <code>selectedValues</code> from {@link Categorical#withParameters(Map selectedValues)} sum to 0
     */
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

    /**
     * @param x T
     * @return log probability at x
     * @throws IllegalArgumentException if <code>selectedValues</code> from {@link Categorical#withParameters(Map selectedValues)} sum to 0
     */
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