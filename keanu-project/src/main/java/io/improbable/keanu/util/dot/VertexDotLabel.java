package io.improbable.keanu.util.dot;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;

import java.util.Objects;

public class VertexDotLabel {
    private final Vertex vertex;
    private String value = "";
    private String annotation = "";
    private String vertexLabel = "";
    private static final String DOT_LABEL_OPENING = "[label=\"";
    private static final String DOT_LABEL_CLOSING = "\"]";

    public VertexDotLabel(Vertex vertex) {
        this.vertex = vertex;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public void setAnnotation(String annotation) {
        this.annotation = annotation;
    }

    public void setVertexLabel(String vertexLabel) {
        this.vertexLabel = vertexLabel;
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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexDotLabel that = (VertexDotLabel) o;
        return Objects.equals(vertex, that.vertex);
    }

    @Override
    public int hashCode() {
        return Objects.hash(vertex);
    }
}
