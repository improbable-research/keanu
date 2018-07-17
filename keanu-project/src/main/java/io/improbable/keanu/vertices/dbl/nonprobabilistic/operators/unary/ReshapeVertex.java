package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class ReshapeVertex extends DoubleUnaryOpVertex {

    private int[] proposedShape;

    public ReshapeVertex(DoubleVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex,
            a -> a.reshape(proposedShape),
            a -> a.reshape(proposedShape)
        );
    }
}
