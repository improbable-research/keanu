package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerReshapeVertex extends IntegerUnaryOpVertex {

    private int[] proposedShape;

    public IntegerReshapeVertex(IntegerVertex inputVertex, int... proposedShape) {
        super(inputVertex.getShape(), inputVertex);
        this.proposedShape = proposedShape;
    }

    /**
     * Returns the supplied vertex with a new shape of the same length
     */
    @Override
    protected IntegerTensor op(IntegerTensor a) {
        return a.reshape(proposedShape);
    }
}
