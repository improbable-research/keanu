package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

public class BoolReshapeVertex extends BoolUnaryOpVertex<BooleanTensor> {

    private int[] proposedShape;

    public BoolReshapeVertex(BoolVertex inputVertex, int... proposedShape) {
        super(inputVertex.getShape(), inputVertex);
        this.proposedShape = proposedShape;
    }

    /**
     * Returns the supplied vertex with a new shape of the same length
     */
    @Override
    protected BooleanTensor op(BooleanTensor booleanTensor) {
        return booleanTensor.reshape(proposedShape);
    }

}
