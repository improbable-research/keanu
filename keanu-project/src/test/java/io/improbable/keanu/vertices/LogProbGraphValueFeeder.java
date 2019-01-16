package io.improbable.keanu.vertices;

/**
 * Prematurely feeds values to a placeholder in a LogProbGraph for testing.
 * Normally values would be fed after LogProbGraph is converted to a computation graph.
 */
public class LogProbGraphValueFeeder {

    public static <T> void feedValue(LogProbGraph logProbGraph, Vertex<T> input, T value) {
        logProbGraph.getPlaceholder(input).setValue(value);
    }

}
