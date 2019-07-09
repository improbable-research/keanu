package io.improbable.keanu.vertices;

public interface VertexBinaryOp<L extends IVertex, R extends IVertex> {

    L getLeft();

    R getRight();
}
