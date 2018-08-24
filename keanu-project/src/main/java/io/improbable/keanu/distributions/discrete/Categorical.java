package io.improbable.keanu.distributions.discrete;

import java.util.Map;
import java.util.Set;

import io.improbable.keanu.distributions.Distribution;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
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

    @Override
    public DoubleTensor computeKLDivergence(Distribution<T> q) {
        if (q instanceof Categorical) {
            Map qSelectableValues = ((Categorical) q).selectableValues;
            if ((qSelectableValues.keySet().containsAll(this.selectableValues.keySet()))) {
                DoubleTensor sum = Nd4jDoubleTensor.scalar(0);
                for (T key : selectableValues.keySet()) {
                    DoubleTensor pProb = selectableValues.get(key);
                    DoubleTensor qProb = (DoubleTensor) qSelectableValues.get(key);

                    sum.plusInPlace(pProb.times(pProb.div(qProb).logInPlace()));
                }

                return sum;
            } else {
                throw new IllegalArgumentException("q must have wider support than p");
            }
        } else {
            return Distribution.super.computeKLDivergence(q);
        }
    }

    private Class<?> getType() {
        Set keys = selectableValues.keySet();
        if (keys.isEmpty()) {
            return null;
        } else {
            return keys.iterator().next().getClass();
        }
    }
}
