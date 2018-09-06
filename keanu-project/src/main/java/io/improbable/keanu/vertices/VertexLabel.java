package io.improbable.keanu.vertices;

import java.util.Objects;

public class VertexLabel {
    private final String namespace;
    private final String label;

    public VertexLabel(String label) {
        this(null, label);
    }

    public VertexLabel(String namespace, String label) {
        this.namespace = namespace;
        this.label = label;
    }

    @Override
    public String toString() {
        return String.format("%s:%s", namespace, label);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexLabel that = (VertexLabel) o;
        return Objects.equals(namespace, that.namespace) &&
            Objects.equals(label, that.label);
    }

    @Override
    public int hashCode() {

        return Objects.hash(namespace, label);
    }

    public VertexLabel inNamespace(String newNamespace) {
        return new VertexLabel(newNamespace, this.label);
    }
}
