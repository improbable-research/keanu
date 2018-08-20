package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ReshapeVertex extends DoubleUnaryOpVertex {

    public ReshapeVertex(DoubleVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.reshape(getShape());
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.reshape(getShape());
    }
}
