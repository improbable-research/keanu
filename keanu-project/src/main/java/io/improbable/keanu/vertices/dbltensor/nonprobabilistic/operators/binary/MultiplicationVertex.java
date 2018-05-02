package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.scaler.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;

public class MultiplicationVertex extends BinaryOpVertex {

    public MultiplicationVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    protected DualNumber getDualNumber(DualNumber aDual, DualNumber bDual) {
        return aDual.multiplyBy(bDual);
    }

    @Override
    protected Double op(Double a, Double b) {
        return a * b;
    }
}
