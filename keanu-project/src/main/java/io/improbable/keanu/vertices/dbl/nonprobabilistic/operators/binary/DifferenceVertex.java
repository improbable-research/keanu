package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class DifferenceVertex extends DoubleBinaryOpVertex {

    public DifferenceVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        final DualNumber aDual = a.getDualNumber();
        final DualNumber bDual = b.getDualNumber();
        return aDual.subtract(bDual);
    }

    protected Double op(Double a, Double b) {
        return a - b;
    }
}
