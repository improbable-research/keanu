package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

public class BoolReshapeVertex extends BoolUnaryOpVertex<BooleanTensor> {

    public BoolReshapeVertex(BoolVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex, a -> a.reshape(proposedShape));
    }
}
