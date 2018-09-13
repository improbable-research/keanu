package io.improbable.keanu.vertices;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

import io.improbable.keanu.tensor.Tensor;

public class SimpleVertexDictionary implements VertexDictionary {

    private final Map<VertexLabel, Vertex<?>> dictionary;

    private SimpleVertexDictionary(Map<VertexLabel, Vertex<?>> dictionary) {
        this.dictionary = dictionary;
    }

    @Override
    public <V extends Vertex<? extends Tensor<?>>> V get(VertexLabel label) {
        return (V) dictionary.get(label);
    }

    public static VertexDictionary backedBy(Map<VertexLabel, Vertex<?>> dictionary) {
        return new SimpleVertexDictionary(dictionary);
    }

    public static VertexDictionary of(Vertex<?>... vertices) {
        Map<VertexLabel, Vertex<?>> dictionary = Arrays.stream(vertices).collect(Collectors.toMap(Vertex::getLabel, v -> v));
        return backedBy(dictionary);
    }
}
