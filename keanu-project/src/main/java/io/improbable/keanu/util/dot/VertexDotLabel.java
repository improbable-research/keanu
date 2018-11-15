package io.improbable.keanu.util.dot;

import io.improbable.keanu.vertices.Vertex;

import java.util.Objects;

public class VertexDotLabel {
    private Vertex vertex;
    private String value = "";
    private String annotation = "";
    private String vertexLabel = "";
    private static final String DOT_LABEL_OPENING = "[label=\"";
    private static final String DOT_LABEL_CLOSING = "\"]";

    public VertexDotLabel(Vertex vertex) {
        this.vertex = vertex;
    }

    public enum VertexDotLabelType{
        VALUE, CLASS_NAME, ANNOTATION_LABEL, VERTEX_LABEL
    }

    public void setDotLabel(VertexDotLabelType labelType, String label) {
        switch (labelType) {
            case VALUE:
                value = label;
            case VERTEX_LABEL:
                vertexLabel = label;
            case ANNOTATION_LABEL:
                annotation = label;
        }
    }

    public String inDotFormat() {
        if (!value.isEmpty()) {
            return vertex.getId().hashCode() + DOT_LABEL_OPENING + value + DOT_LABEL_CLOSING;
        }
        if (!vertexLabel.isEmpty()) {
            return vertex.getId().hashCode() + DOT_LABEL_OPENING + vertexLabel + DOT_LABEL_CLOSING;
        }
        if (!annotation.isEmpty()) {
            return vertex.getId().hashCode() + DOT_LABEL_OPENING + annotation + DOT_LABEL_CLOSING;
        }
        return vertex.getId().hashCode() + DOT_LABEL_OPENING + vertex.getClass().getSimpleName() + DOT_LABEL_CLOSING;
    }

    public boolean equals(Object o) {
        return o.hashCode() == this.hashCode();
    }

    public int hashCode() {
        return Objects.hash(vertex);
    }
}
