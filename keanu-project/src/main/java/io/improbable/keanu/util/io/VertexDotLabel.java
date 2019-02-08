package io.improbable.keanu.util.io;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;

import java.util.Objects;
import java.util.Optional;

public class VertexDotLabel {
    private final Vertex vertex;
    private String value = "";
    private String annotation = "";
    private String vertexLabel = "";
    private static final String DOT_LABEL_OPENING = "[label=\"";
    private static final String DOT_LABEL_CLOSING = "\"]";

    public VertexDotLabel(Vertex vertex) {
        this.vertex = vertex;
        if (vertex.getLabel() != null) {
            vertexLabel = vertex.getLabel().getUnqualifiedName();
        }
        DisplayInformationForOutput vertexAnnotation = vertex.getClass().getAnnotation(DisplayInformationForOutput.class);
        if (vertexAnnotation != null && !vertexAnnotation.displayName().isEmpty()) {
            annotation = vertexAnnotation.displayName();
        }
    }

    public void setValue(String value) {
        this.value = value;
    }

    public String inDotFormat() {
        // Output value if value is set, but also add some descriptive info.
        if (!value.isEmpty()) {
            StringBuilder dotLabel = new StringBuilder();
            dotLabel.append(vertex.getId().hashCode()).append(DOT_LABEL_OPENING).append(value);
            getDescriptiveInfoForVertexWithValue().ifPresent(info -> dotLabel.append(" (").append(info).append(")"));
            return dotLabel.append(DOT_LABEL_CLOSING).toString();
        }
        return vertex.getId().hashCode() + DOT_LABEL_OPENING + getDescriptiveInfo() + DOT_LABEL_CLOSING;
    }

    private Optional<String> getDescriptiveInfoForVertexWithValue() {
        if (!vertexLabel.isEmpty()) {
            return Optional.of(vertexLabel);
        }
        if (!(vertex instanceof ConstantVertex)) {
            return Optional.of(getAnnotationIfPresentElseSimpleName());
        }
        return Optional.empty();
    }

    private String getDescriptiveInfo() {
        if (!vertexLabel.isEmpty()) {
            return vertexLabel;
        }
        return getAnnotationIfPresentElseSimpleName();
    }

    private String getAnnotationIfPresentElseSimpleName() {
        return annotation.isEmpty() ? vertex.getClass().getSimpleName() : annotation;
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
