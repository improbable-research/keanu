package io.improbable.keanu.vertices;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public class VertexLabel {
    private static final char NAMESPACE_SEPARATOR = '.';
    private final List<String> namespace;
    private final String name;

    public VertexLabel(String outer, String... inner) {
        if (inner.length <= 0) {
            this.namespace = ImmutableList.of();
            this.name = outer;
        } else {
            this.namespace = ImmutableList.<String>builder()
                .add(outer)
                .add(Arrays.copyOf(inner, inner.length - 1))
                .build();
            this.name = inner[inner.length - 1];
        }
    }

    public VertexLabel(List<String> namespace, String name) {
        this.namespace = ImmutableList.copyOf(namespace);
        this.name = name;
    }

    public boolean isInNamespace(String... namespace) {

        if (namespace.length > this.namespace.size()) {
            return false;
        }

        for (int i = 0; i < namespace.length; i++) {
            if (!this.namespace.get(i).equals(namespace[i])) {
                return false;
            }
        }
        return true;
    }

    public VertexLabel withExtraNamespace(String topLevelNamespace) {
        List<String> newNamespace = ImmutableList.<String>builder().add(topLevelNamespace).addAll(namespace).build();
        return new VertexLabel(newNamespace, this.name);
    }

    public VertexLabel withoutOuterNamespace() {
        List<String> reducedNamespace = namespace.subList(1, namespace.size());
        return new VertexLabel(reducedNamespace, this.name);
    }

    public Optional<String> getOuterNamespace() {
        return namespace.stream().findFirst();
    }

    public String getUnqualifiedName() {
        return name;
    }

    public String getQualifiedName() {
        ImmutableList<String> names = ImmutableList.<String>builder().addAll(namespace).add(name).build();
        return Joiner.on(NAMESPACE_SEPARATOR).join(names);
    }

    @Override
    public String toString() {
        return getQualifiedName();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexLabel that = (VertexLabel) o;
        return Objects.equals(name, that.name) &&
            Objects.equals(namespace, that.namespace);
    }

    @Override
    public int hashCode() {

        return Objects.hash(namespace, name);
    }
}
