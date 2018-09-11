package io.improbable.keanu.vertices;

import java.util.Map;

import io.improbable.keanu.tensor.Tensor;

public interface VertexDictionary {
    <V extends Vertex<? extends Tensor<?>>> V get(VertexLabel label);

    static VertexDictionary backedBy(Map<VertexLabel, Vertex<?>> map) {
        return SimpleVertexDictionary.backedBy(map);
    }

    static VertexDictionary of(Vertex... vertices) {
        return SimpleVertexDictionary.of(vertices);
    }
}
