package io.improbable.keanu.vertices;

public interface VertexBinaryOp<L extends Vertex, R extends Vertex> {

    L getLeft();

    R getRight();
}
