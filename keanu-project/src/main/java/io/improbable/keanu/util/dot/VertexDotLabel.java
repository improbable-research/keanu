package io.improbable.keanu.util.dot;

import io.improbable.keanu.vertices.ConstantVertex;
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
                break;
            case VERTEX_LABEL:
                vertexLabel = label;
                break;
            case ANNOTATION_LABEL:
                annotation = label;
        }
    }

    public String inDotFormat() {
        // Output value if value is set, but also add some descriptive info for non-constant vertices.
        if (!value.isEmpty()) {
            String dotLabel = vertex.getId().hashCode() + DOT_LABEL_OPENING + value;
            if (!(vertex instanceof ConstantVertex)) {
                dotLabel += " (" + getDescriptiveInfo() + ")";
            }
            return dotLabel + DOT_LABEL_CLOSING;
        }
        return vertex.getId().hashCode() + DOT_LABEL_OPENING + getDescriptiveInfo() + DOT_LABEL_CLOSING;
    }

    public String getDescriptiveInfo() {
        if (!vertexLabel.isEmpty()) {
            return vertexLabel;
        }
        if (!annotation.isEmpty()) {
            return annotation;
        }
        return vertex.getClass().getSimpleName();
    }

    public boolean equals(Object o) {
        return o.hashCode() == this.hashCode();
    }

    public int hashCode() {
        return Objects.hash(vertex);
    }
}
