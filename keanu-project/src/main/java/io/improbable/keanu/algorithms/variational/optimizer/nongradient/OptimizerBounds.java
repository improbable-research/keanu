package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.HashMap;
import java.util.Map;

public class OptimizerBounds {

    @Value
    private static class VertexBounds {
        DoubleTensor min;
        DoubleTensor max;
    }

    private Map<Vertex<? extends DoubleTensor>, VertexBounds> vertexBounds = new HashMap<>();

    public void addBound(Vertex<? extends DoubleTensor> vertex, DoubleTensor min, DoubleTensor max) {
        DoubleTensor minDup = min.duplicate();
        DoubleTensor maxDup = max.duplicate();

        vertexBounds.put(vertex, new VertexBounds(minDup, maxDup));
    }

    public void addBound(Vertex<? extends DoubleTensor> vertex, double min, DoubleTensor max) {
        addBound(vertex, DoubleTensor.scalar(min), max);
    }

    public void addBound(Vertex<? extends DoubleTensor> vertex, DoubleTensor min, double max) {
        addBound(vertex, min, DoubleTensor.scalar(max));
    }

    public void addBound(Vertex<? extends DoubleTensor> vertex, double min, double max) {
        addBound(vertex, DoubleTensor.scalar(min), DoubleTensor.scalar(max));
    }

    public boolean hasBound(Vertex<? extends DoubleTensor> vertex) {
        return vertexBounds.containsKey(vertex);
    }

    public DoubleTensor getLower(Vertex<? extends DoubleTensor> vertex) {
        return vertexBounds.get(vertex).getMin();
    }

    public DoubleTensor getUpper(Vertex<? extends DoubleTensor> vertex) {
        return vertexBounds.get(vertex).getMax();
    }

}
