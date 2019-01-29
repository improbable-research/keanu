package io.improbable.keanu.util.io;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.lang3.StringUtils;

import java.util.Map;
import java.util.Objects;

public class VertexDotLabel {
    private static final String DOT_LABEL_OPENING = "[label=\"";
    private static final String DOT_LABEL_CLOSING = "\"]";
    private static final String DOT_FIELD_OPENING = " [";
    private static final String DOT_FIELD_SEPARATOR = "=";
    private static final String DOT_FIELD_CLOSING = "]";
    private final Vertex vertex;
    private String value = "";
    private String annotation = "";
    private String vertexLabel = "";

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

    public String inDotFormat(DotDecorator decorator) {
        String dotLabel = vertex.getId().hashCode() + DOT_LABEL_OPENING;
        if (this.value.length() > 0) dotLabel += this.value;
        else dotLabel += StringUtils.join(decorator.labelVertex(vertex), ", ");
        dotLabel += DOT_LABEL_CLOSING;
        Map<String, String> fields = decorator.getExtraVertexFields(vertex);
        for (Map.Entry<String, String> e : fields.entrySet()) {
            dotLabel += DOT_FIELD_OPENING;
            dotLabel += e.getKey();
            dotLabel += DOT_FIELD_SEPARATOR;
            dotLabel += e.getValue();
            dotLabel += DOT_FIELD_CLOSING;
        }
        return dotLabel;
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
