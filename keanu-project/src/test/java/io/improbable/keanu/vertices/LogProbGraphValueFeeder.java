package io.improbable.keanu.vertices;

/**
 * Prematurely feeds values to a placeholder in a LogProbGraph for testing.
 * Normally values would be fed after LogProbGraph is converted to a computation graph.
 */
public class LogProbGraphValueFeeder {

    public static <T> void feedValue(LogProbGraph logProbGraph, IVertex<T> input, T value) {
        IVertex<T> placeholderVertex = logProbGraph.getPlaceholder(input);
        placeholderVertex.setValue(value);
    }

    public static <T> void feedValueAndCascade(LogProbGraph logProbGraph, IVertex<T> input, T value) {
        IVertex<T> placeholderVertex = logProbGraph.getPlaceholder(input);
        placeholderVertex.setAndCascade(value);
    }
}
