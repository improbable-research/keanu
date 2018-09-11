package io.improbable.keanu.vertices;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

import io.improbable.keanu.tensor.Tensor;

public class SimpleVertexDictionary implements VertexDictionary {
    private final Map<VertexLabel, Vertex<?>> map;

    private SimpleVertexDictionary(Map<VertexLabel, Vertex<?>> map) {
        this.map = map;
    }

    @Override
    public <V extends Vertex<? extends Tensor<?>>> V get(VertexLabel label) {
        return (V) map.get(label);
    }

    public static VertexDictionary backedBy(Map<VertexLabel, Vertex<?>> map) {
        return new SimpleVertexDictionary(map);
    }

    public static VertexDictionary of(Vertex<?>... vertices) {
        Map<VertexLabel, Vertex<?>> map = Arrays.stream(vertices).collect(Collectors.toMap(Vertex::getLabel, v -> v));
        return backedBy(map);
    }
}
