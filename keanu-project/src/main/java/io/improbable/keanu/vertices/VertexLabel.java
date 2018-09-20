package io.improbable.keanu.vertices;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

public class VertexLabel {
    private static final char NAMESPACE_SEPARATOR = '.';
    private final String name;
    private final List<String> namespace;

    public VertexLabel(String name, String... namespace) {
        this(name, ImmutableList.copyOf(namespace));
    }

    public VertexLabel(String name, List<String> namespace) {
        this.name = name;
        this.namespace = ImmutableList.copyOf(namespace);
    }

    public VertexLabel withExtraNamespace(String topLevelNamespace) {
        List<String> newNamespace = ImmutableList.<String>builder().addAll(namespace).add(topLevelNamespace).build();
        return new VertexLabel(this.name, newNamespace);
    }

    public VertexLabel withoutOuterNamespace() {
        try {
            List<String> reducedNamespace = namespace.subList(0, namespace.size() - 1);
            return new VertexLabel(this.name, reducedNamespace);
        } catch (IndexOutOfBoundsException e) {
            throw new VertexLabelException("There is no namespace to remove", e);
        }
    }

    public Optional<String> getOuterNamespace() {
        try {
            return Optional.of(namespace.get(namespace.size() - 1));
        } catch(IndexOutOfBoundsException e) {
            return Optional.empty();
        }
    }

    public String getUnqualifiedName() {
        return name;
    }

    @Override
    public String toString() {
        ImmutableList<String> names = ImmutableList.<String>builder().add(name).addAll(namespace).build();
        return Joiner.on(NAMESPACE_SEPARATOR).join(Lists.reverse(names));
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

        return Objects.hash(name, namespace);
    }
}
