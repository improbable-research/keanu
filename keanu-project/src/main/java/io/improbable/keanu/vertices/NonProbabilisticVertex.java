package io.improbable.keanu.vertices;

public interface NonProbabilisticVertex<T, VERTEX extends Vertex<T, VERTEX>> extends Vertex<T, VERTEX>, NonProbabilistic<T> {
}
