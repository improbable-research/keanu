package io.improbable.keanu.vertices;

import com.google.common.collect.ImmutableMap;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

public class SimpleVertexDictionary implements VertexDictionary {

    private final Map<VertexLabel, IVertex<?>> dictionary;

    private SimpleVertexDictionary(Map<VertexLabel, IVertex<?>> dictionary) {
        this.dictionary = dictionary;
    }

    public static SimpleVertexDictionary combine(SimpleVertexDictionary dictionary, SimpleVertexDictionary dictionary2) {
        ImmutableMap<VertexLabel, IVertex<?>> combinedDictionary = ImmutableMap.<VertexLabel, IVertex<?>>builder()
            .putAll(dictionary.dictionary)
            .putAll(dictionary2.dictionary)
            .build();

        return SimpleVertexDictionary.backedBy(combinedDictionary);
    }

    @Override
    public <V extends IVertex<?>> V get(VertexLabel label) {
        return (V) dictionary.get(label);
    }

    @Override
    public VertexDictionary withExtraEntries(Map<VertexLabel, IVertex<?>> extraEntries) {
        return SimpleVertexDictionary.backedBy(dictionary, extraEntries);
    }

    public static SimpleVertexDictionary backedBy(Map<VertexLabel, IVertex<?>> dictionary) {
        return new SimpleVertexDictionary(dictionary);
    }

    public static SimpleVertexDictionary backedBy(Map<VertexLabel, IVertex<?>> first, Map<VertexLabel, IVertex<?>> second) {
        return SimpleVertexDictionary.backedBy(
            ImmutableMap.<VertexLabel, IVertex<?>>builder()
                .putAll(first)
                .putAll(second)
                .build());
    }

    public static SimpleVertexDictionary of(IVertex<?>... vertices) {
        Map<VertexLabel, IVertex<?>> dictionary = Arrays.stream(vertices).collect(Collectors.toMap(IVertex::getLabel, v -> v));
        return backedBy(dictionary);
    }
}
