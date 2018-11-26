package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.HashMap;
import java.util.Map;

public class OptimizerBounds {

    @Value
    private static class VariableBounds {
        DoubleTensor min;
        DoubleTensor max;
    }

    private Map<String, VariableBounds> variableBounds = new HashMap<>();

    public void addBound(String variable, DoubleTensor min, DoubleTensor max) {
        DoubleTensor minDup = min.duplicate();
        DoubleTensor maxDup = max.duplicate();

        variableBounds.put(variable, new VariableBounds(minDup, maxDup));
    }

    public void addBound(String variable, double min, DoubleTensor max) {
        addBound(variable, DoubleTensor.scalar(min), max);
    }

    public void addBound(String variable, DoubleTensor min, double max) {
        addBound(variable, min, DoubleTensor.scalar(max));
    }

    public void addBound(String variable, double min, double max) {
        addBound(variable, DoubleTensor.scalar(min), DoubleTensor.scalar(max));
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

    public void addBound(Vertex<?> variable, DoubleTensor min, DoubleTensor max) {
        addBound(variable.getUniqueStringReference(), min, max);
    }

    public void addBound(Vertex<?> variable, double min, DoubleTensor max) {
        addBound(variable.getUniqueStringReference(), DoubleTensor.scalar(min), max);
    }

    public void addBound(Vertex<?> variable, DoubleTensor min, double max) {
        addBound(variable.getUniqueStringReference(), min, DoubleTensor.scalar(max));
    }

    public void addBound(Vertex<?> variable, double min, double max) {
        addBound(variable.getUniqueStringReference(), DoubleTensor.scalar(min), DoubleTensor.scalar(max));
    }

    public boolean hasBound(Vertex<?> variable) {
        return hasBound(variable.getUniqueStringReference());
    }

    public DoubleTensor getLower(Vertex<?> variable) {
        return getLower(variable.getUniqueStringReference());
    }

    public DoubleTensor getUpper(Vertex<?> variable) {
        return getUpper(variable.getUniqueStringReference());
    }

}
