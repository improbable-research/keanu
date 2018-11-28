package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.algorithms.variational.optimizer.nongradient.KeanuProbabilisticGraph.getUniqueStringReference;

public class OptimizerBounds {

    @Value
    private static class VariableBounds {
        DoubleTensor min;
        DoubleTensor max;
    }

    private Map<String, VariableBounds> variableBounds = new HashMap<>();

    public OptimizerBounds addBound(String variable, DoubleTensor min, DoubleTensor max) {
        DoubleTensor minDup = min.duplicate();
        DoubleTensor maxDup = max.duplicate();

        variableBounds.put(variable, new VariableBounds(minDup, maxDup));
        return this;
    }

    public OptimizerBounds addBound(String variable, double min, DoubleTensor max) {
        addBound(variable, DoubleTensor.scalar(min), max);
        return this;
    }

    public OptimizerBounds addBound(String variable, DoubleTensor min, double max) {
        addBound(variable, min, DoubleTensor.scalar(max));
        return this;
    }

    public OptimizerBounds addBound(String variable, double min, double max) {
        addBound(variable, DoubleTensor.scalar(min), DoubleTensor.scalar(max));
        return this;
    }

    public boolean hasBound(String variable) {
        return variableBounds.containsKey(variable);
    }

    public DoubleTensor getLower(String variable) {
        return variableBounds.get(variable).getMin();
    }

    public DoubleTensor getUpper(String variable) {
        return variableBounds.get(variable).getMax();
    }

    public OptimizerBounds addBound(Vertex<?> variable, DoubleTensor min, DoubleTensor max) {
        addBound(getUniqueStringReference(variable), min, max);
        return this;
    }

    public OptimizerBounds addBound(Vertex<?> variable, double min, DoubleTensor max) {
        addBound(getUniqueStringReference(variable), DoubleTensor.scalar(min), max);
        return this;
    }

    public OptimizerBounds addBound(Vertex<?> variable, DoubleTensor min, double max) {
        addBound(getUniqueStringReference(variable), min, DoubleTensor.scalar(max));
        return this;
    }

    public OptimizerBounds addBound(Vertex<?> variable, double min, double max) {
        addBound(getUniqueStringReference(variable), DoubleTensor.scalar(min), DoubleTensor.scalar(max));
        return this;
    }

    public boolean hasBound(Vertex<?> variable) {
        return hasBound(getUniqueStringReference(variable));
    }

    public DoubleTensor getLower(Vertex<?> variable) {
        return getLower(getUniqueStringReference(variable));
    }

    public DoubleTensor getUpper(Vertex<?> variable) {
        return getUpper(getUniqueStringReference(variable));
    }

}
