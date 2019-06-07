package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class VariableValues {

    private VariableValues() {
    }

    public static double dotProduct(Map<? extends VariableReference, DoubleTensor> left, Map<? extends VariableReference, DoubleTensor> right) {
        double dotProduct = 0.0;
        for (Entry<? extends VariableReference, DoubleTensor> entry : left.entrySet()) {
            dotProduct += entry.getValue().times(right.get(entry.getKey())).sum();
        }
        return dotProduct;
    }

    public static Map<VariableReference, DoubleTensor> pow(Map<VariableReference, DoubleTensor> values, double exponent) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : values.entrySet()) {
            result.put(entry.getKey(), entry.getValue().pow(exponent));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> divide(Map<VariableReference, DoubleTensor> left, double right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : left.entrySet()) {
            result.put(entry.getKey(), entry.getValue().div(right));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> times(Map<VariableReference, DoubleTensor> left, double right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : left.entrySet()) {
            result.put(entry.getKey(), entry.getValue().times(right));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> times(Map<VariableReference, DoubleTensor> left, Map<VariableReference, DoubleTensor> right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : left.entrySet()) {
            result.put(entry.getKey(), entry.getValue().times(right.get(entry.getKey())));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> add(Map<VariableReference, DoubleTensor> left, Map<VariableReference, DoubleTensor> right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : left.entrySet()) {
            result.put(entry.getKey(), entry.getValue().plus(right.get(entry.getKey())));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> withShape(double value, Map<VariableReference, DoubleTensor> shapeLike) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (Entry<VariableReference, DoubleTensor> entry : shapeLike.entrySet()) {
            result.put(entry.getKey(), DoubleTensor.create(value, entry.getValue().getShape()));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> zeros(Map<VariableReference, DoubleTensor> shapeLike) {
        return withShape(0.0, shapeLike);
    }

    public static Map<VariableReference, DoubleTensor> ones(Map<VariableReference, DoubleTensor> shapeLike) {
        return withShape(1.0, shapeLike);
    }

}
