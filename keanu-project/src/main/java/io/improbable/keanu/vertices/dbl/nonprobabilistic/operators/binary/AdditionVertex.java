package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class AdditionVertex extends DoubleBinaryOpVertex {

    public AdditionVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        final DualNumber aDual = a.getDualNumber();
        final DualNumber bDual = b.getDualNumber();
        return aDual.add(bDual);
    }

    @Override
    protected Double op(Double a, Double b) {
        return a + b;
    }
}
