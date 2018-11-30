package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.Value;

import java.util.HashMap;
import java.util.Map;

public class OptimizerBounds {

    @Value
    private static class VariableBounds {
        DoubleTensor min;
        DoubleTensor max;
    }

    private Map<VariableReference, VariableBounds> variableBounds = new HashMap<>();

    public OptimizerBounds addBound(VariableReference variable, DoubleTensor min, DoubleTensor max) {
        DoubleTensor minDup = min.duplicate();
        DoubleTensor maxDup = max.duplicate();

        variableBounds.put(variable, new VariableBounds(minDup, maxDup));
        return this;
    }

    public OptimizerBounds addBound(VariableReference variable, double min, DoubleTensor max) {
        addBound(variable, DoubleTensor.scalar(min), max);
        return this;
    }

    public OptimizerBounds addBound(VariableReference variable, DoubleTensor min, double max) {
        addBound(variable, min, DoubleTensor.scalar(max));
        return this;
    }

    public OptimizerBounds addBound(VariableReference variable, double min, double max) {
        addBound(variable, DoubleTensor.scalar(min), DoubleTensor.scalar(max));
        return this;
    }

    public boolean hasBound(VariableReference variable) {
        return variableBounds.containsKey(variable);
    }

    public DoubleTensor getLower(VariableReference variable) {
        return variableBounds.get(variable).getMin();
    }

    public DoubleTensor getUpper(VariableReference variable) {
        return variableBounds.get(variable).getMax();
    }
}
