package io.improbable.keanu.vertices;

/**
 * Prematurely feeds values to LogProbGraph for testing
 */
public class LogProbGraphValueFeeder {

    public static <T> void feedValue(LogProbGraph logProbGraph, Vertex<T> vertex, T value) {
        logProbGraph.getPlaceHolder(vertex).setValue(value);
    }

}
