package io.improbable.keanu.vertices;

import java.util.Objects;

public class VertexLabel {
    private final String label;

    public VertexLabel(String label) {
        this.label = label;
    }

    @Override
    public String toString() {
        return label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexLabel that = (VertexLabel) o;
        return Objects.equals(label, that.label);
    }

    @Override
    public int hashCode() {

        return Objects.hash(label);
    }
}
