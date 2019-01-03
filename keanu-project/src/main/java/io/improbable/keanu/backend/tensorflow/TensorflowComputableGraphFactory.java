package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraphConverter;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Set;

public class TensorflowComputableGraphFactory {

    public static TensorflowComputableGraph convert(Set<Vertex> vertices) {
        return convert(vertices, new TensorflowGraphBuilder());
    }

    public static TensorflowComputableGraph convert(Collection<? extends Vertex> vertices, TensorflowGraphBuilder graphBuilder) {
        return ComputableGraphConverter.convert(vertices, graphBuilder);
    }
}
