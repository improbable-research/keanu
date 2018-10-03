package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

/** Returns the supplied vertex with a new shape of the same length */
public class BoolReshapeVertex extends BoolUnaryOpVertex<BooleanTensor> {

    public BoolReshapeVertex(BoolVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.reshape(getShape());
    }
}
