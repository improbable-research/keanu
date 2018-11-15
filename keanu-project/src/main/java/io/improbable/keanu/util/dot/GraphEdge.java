package io.improbable.keanu.util.dot;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.lang3.StringUtils;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Class representing a graph edge.
 */
public class GraphEdge {

    private Vertex parentVertex;
    private Vertex childVertex;
    private Set<String> dotLabels = new HashSet<>();
    private static final String DOT_LABEL_OPENING = " [label=";
    private static final String DOT_LABEL_CLOSING = "]";

    public GraphEdge(Vertex v1, Vertex v2) {
        if (v1.getId().compareTo(v2.getId()) < 0) {
            parentVertex = v1;
            childVertex = v2;
        } else {
            parentVertex = v2;
            childVertex = v1;
        }
    }

    public void appendToDotLabel(Set<String> dotLabel) {
        dotLabels.addAll(dotLabel);
    }

    public void appendToDotLabel(String dotLabel) {
        dotLabels.add(dotLabel);
    }

    public Set<String> getDotLabels() { return dotLabels;}

    public Vertex getParentVertex() { return parentVertex;}

    public Vertex getChildVertex() { return childVertex;}

    // Returns a string representing this edge in a DOT format.
    public String inDotFormat() {
        String dotOutput = "<" + parentVertex.hashCode() + "> -> <" + childVertex.hashCode() + ">";
        if (!dotLabels.isEmpty()) {
            dotOutput += DOT_LABEL_OPENING;
            dotOutput += StringUtils.join(dotLabels, ", ");
            dotOutput += DOT_LABEL_CLOSING;
        }

        return dotOutput;
    }

    public boolean equals(Object o) {
        return o.hashCode() == this.hashCode();
    }

    public int hashCode() {
        return Objects.hash(parentVertex, childVertex);
    }
}
