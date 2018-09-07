package io.improbable.keanu.vertices;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;

public class VertexLabel {
    private static final char NAMESPACE_SEPARATOR = '.';
    private static final char NAME_PREPENDER = ':';
    private final String name;
    private final List<String> namespace;

    public VertexLabel(String name, String... namespace) {
        this.name = name;
        this.namespace = ImmutableList.copyOf(namespace);
    }

    public VertexLabel inNamespace(String topLevelNamespace) {
        List<String> newNamespace = ImmutableList.<String>builder().addAll(namespace).add(topLevelNamespace).build();
        return new VertexLabel(this.name, newNamespace.toArray(new String[0]));
    }

    public VertexLabel dropNamespace() throws VertexLabelException {
        try {
            List<String> reducedNamespace = namespace.subList(0, namespace.size()-1);
            return new VertexLabel(this.name, reducedNamespace.toArray(new String[0]));
        } catch (IndexOutOfBoundsException e) {
            throw new VertexLabelException("There is no namespace to drop", e);
        }
    }

    @Override
    public String toString() {
        ArrayList reversedNamespace = new ArrayList(namespace);
        Collections.reverse(reversedNamespace);
        String namespaceString = Joiner.on(NAMESPACE_SEPARATOR).join(reversedNamespace);
        return String.format("%s%c%s", namespaceString, NAME_PREPENDER, name);
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
