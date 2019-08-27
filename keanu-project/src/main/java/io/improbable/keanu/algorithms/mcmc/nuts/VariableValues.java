package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class VariableValues {

    private VariableValues() {
    }

    public static double dotProduct(Map<? extends VariableReference, DoubleTensor> left, Map<? extends VariableReference, DoubleTensor> right) {
        double dotProduct = 0.0;
        for (VariableReference v : left.keySet()) {
            dotProduct += left.get(v).times(right.get(v)).sumNumber();
        }
        return dotProduct;
    }

    public static Map<VariableReference, DoubleTensor> pow(Map<VariableReference, DoubleTensor> values, double exponent) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : values.keySet()) {
            result.put(v, values.get(v).pow(exponent));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> divide(Map<VariableReference, DoubleTensor> left, double right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : left.keySet()) {
            result.put(v, left.get(v).div(right));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> times(Map<VariableReference, DoubleTensor> left, double right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : left.keySet()) {
            result.put(v, left.get(v).times(right));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> times(Map<VariableReference, DoubleTensor> left, Map<VariableReference, DoubleTensor> right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : left.keySet()) {
            result.put(v, left.get(v).times(right.get(v)));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> add(Map<VariableReference, DoubleTensor> left, Map<VariableReference, DoubleTensor> right) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : left.keySet()) {
            result.put(v, left.get(v).plus(right.get(v)));
        }
        return result;
    }

    public static Map<VariableReference, DoubleTensor> withShape(double value, Map<VariableReference, DoubleTensor> shapeLike) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>();
        for (VariableReference v : shapeLike.keySet()) {
            result.put(v, DoubleTensor.create(value, shapeLike.get(v).getShape()));
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
