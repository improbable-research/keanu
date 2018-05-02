package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.scaler.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;

public class DifferenceVertex extends BinaryOpVertex {

    public DifferenceVertex(DoubleTensorVertex a, DoubleTensorVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber getDualNumber(DualNumber aDual, DualNumber bDual) {
        return aDual.subtract(bDual);
    }

    protected Double op(Double a, Double b) {
        return a - b;
    }
}
