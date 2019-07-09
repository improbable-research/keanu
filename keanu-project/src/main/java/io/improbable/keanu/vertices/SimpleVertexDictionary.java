package io.improbable.keanu.vertices;

import com.google.common.collect.ImmutableMap;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

public class SimpleVertexDictionary implements VertexDictionary {

    private final Map<VertexLabel, Vertex<?>> dictionary;

    private SimpleVertexDictionary(Map<VertexLabel, Vertex<?>> dictionary) {
        this.dictionary = dictionary;
    }

    public static SimpleVertexDictionary combine(SimpleVertexDictionary dictionary, SimpleVertexDictionary dictionary2) {
        ImmutableMap<VertexLabel, Vertex<?>> combinedDictionary = ImmutableMap.<VertexLabel, Vertex<?>>builder()
            .putAll(dictionary.dictionary)
            .putAll(dictionary2.dictionary)
            .build();

        return SimpleVertexDictionary.backedBy(combinedDictionary);
    }

    @Override
    public <V extends Vertex<?>> V get(VertexLabel label) {
        return (V) dictionary.get(label);
    }

    @Override
    public VertexDictionary withExtraEntries(Map<VertexLabel, Vertex<?>> extraEntries) {
        return SimpleVertexDictionary.backedBy(dictionary, extraEntries);
    }

    public static SimpleVertexDictionary backedBy(Map<VertexLabel, Vertex<?>> dictionary) {
        return new SimpleVertexDictionary(dictionary);
    }

    public static SimpleVertexDictionary backedBy(Map<VertexLabel, Vertex<?>> first, Map<VertexLabel, Vertex<?>> second) {
        return SimpleVertexDictionary.backedBy(
            ImmutableMap.<VertexLabel, Vertex<?>>builder()
                .putAll(first)
                .putAll(second)
                .build());
    }

    public static SimpleVertexDictionary of(Vertex<?>... vertices) {
        Map<VertexLabel, Vertex<?>> dictionary = Arrays.stream(vertices).collect(Collectors.toMap(Vertex::getLabel, v -> v));
        return backedBy(dictionary);
    }
}
