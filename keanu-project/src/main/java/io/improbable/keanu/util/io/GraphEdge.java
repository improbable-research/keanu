package io.improbable.keanu.util.io;

import io.improbable.keanu.vertices.Vertex;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Class representing a graph edge.
 */
public class GraphEdge {

    private final Vertex parentVertex;
    private final Vertex childVertex;
    private Set<String> labels = new HashSet<>();

    public GraphEdge(Vertex v1, Vertex v2) {
        if (v1.getId().compareTo(v2.getId()) < 0) {
            parentVertex = v1;
            childVertex = v2;
        } else {
            parentVertex = v2;
            childVertex = v1;
        }
    }

    public void appendToLabel(String dotLabel) {
        labels.add(dotLabel);
    }

    public Vertex getParentVertex() {
        return parentVertex;
    }

    public Vertex getChildVertex() {
        return childVertex;
    }

    public Set<String> getLabels() {
        return labels;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphEdge graphEdge = (GraphEdge) o;
        return Objects.equals(parentVertex, graphEdge.parentVertex) &&
            Objects.equals(childVertex, graphEdge.childVertex);
    }

    @Override
    public int hashCode() {

        return Objects.hash(parentVertex, childVertex);
    }
}
