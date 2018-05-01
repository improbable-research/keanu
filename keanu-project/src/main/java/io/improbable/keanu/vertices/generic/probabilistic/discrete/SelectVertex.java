package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.probabilistic.Probabilistic;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

public class SelectVertex<T> extends Probabilistic<T> {

    private final LinkedHashMap<T, DoubleVertex> selectableValues;
    private final Random random;

    public SelectVertex(LinkedHashMap<T, DoubleVertex> selectableValues, Random random) {
        this.selectableValues = selectableValues;
        this.random = random;
        setParents(selectableValues.values());
    }

    public SelectVertex(LinkedHashMap<T, DoubleVertex> selectableValues) {
        this(selectableValues, new Random());
    }

    public LinkedHashMap<T, DoubleVertex> getSelectableValues() {
        return selectableValues;
    }

    @Override
    public T sample() {
        double sumOfProbabilities = getSumOfProbabilities();
        double p = random.nextDouble();
        double sum = 0;

        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }

        T value = null;
        for (Map.Entry<T, DoubleVertex> entry : selectableValues.entrySet()) {
            sum += entry.getValue().getValue() / sumOfProbabilities;
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

    public double logProb(T value) {
        double sumOfProbabilities = getSumOfProbabilities();
        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }
        final double probability = selectableValues.get(value).getValue() / sumOfProbabilities;
        return Math.log(probability);
    }

    @Override
    public Map<String, Double> dLogProb(T value) {
        throw new UnsupportedOperationException();
    }

    private double getSumOfProbabilities() {
        double sumP = 0.0;
        for (DoubleVertex p : selectableValues.values()) {
            sumP += p.getValue();
        }
        return sumP;
    }
}
