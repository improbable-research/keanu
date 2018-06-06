package io.improbable.keanu.vertices.generic.probabilistic.discrete;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.probabilistic.Probabilistic;

import java.util.LinkedHashMap;
import java.util.Map;

public class SelectVertex<T> extends Probabilistic<T, Tensor<T>> {

    private final Map<T, DoubleVertex> selectableValues;

    public SelectVertex(Map<T, DoubleVertex> selectableValues) {
        this.selectableValues = defensiveCopy(selectableValues);
        setParents(this.selectableValues.values());
    }

    private Map<T, DoubleVertex> defensiveCopy(Map<T, DoubleVertex> selectableValues) {
        LinkedHashMap<T, DoubleVertex> copy = new LinkedHashMap<>();
        for (Map.Entry<T, DoubleVertex> entry : selectableValues.entrySet()) {
            if (!TensorShape.isScalar(entry.getValue().getShape())) {
                throw new IllegalArgumentException("Selected probability must be scalar");
            }
            copy.put(entry.getKey(), entry.getValue());
        }
        return copy;
    }

    public Map<T, DoubleVertex> getSelectableValues() {
        return selectableValues;
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
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

        return Tensor.scalar(value);
    }

    public double logProbOf(T value) {
        return logProb(Tensor.scalar(value));
    }

    @Override
    public double logProb(Tensor<T> value) {
        double sumOfProbabilities = getSumOfProbabilities();
        if (sumOfProbabilities == 0.0) {
            throw new IllegalArgumentException("Cannot sample from a zero probability setup.");
        }
        final double probability = selectableValues.get(value.scalar()).getValue().scalar() / sumOfProbabilities;
        return Math.log(probability);
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(Tensor<T> value) {
        throw new UnsupportedOperationException();
    }

    private double getSumOfProbabilities() {
        double sumP = 0.0;
        for (DoubleVertex p : selectableValues.values()) {
            sumP += p.getValue().scalar();
        }
        return sumP;
    }
}
