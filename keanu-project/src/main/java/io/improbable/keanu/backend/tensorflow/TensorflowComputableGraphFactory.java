package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.vertices.Vertex;

import java.util.Set;

public class TensorflowComputableGraphFactory {

    public static TensorflowComputableGraph convert(Set<Vertex> vertices) {
        TensorflowGraphBuilder graphBuilder = new TensorflowGraphBuilder();
        graphBuilder.convert(vertices);
        return graphBuilder.build();
    }
}
