package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;


public class DivisionVertex extends DoubleBinaryOpVertex {

    public DivisionVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber aDual = a.getDualNumber();
        DualNumber bDual = b.getDualNumber();
        return aDual.divideBy(bDual);
    }

    protected Double op(Double a, Double b) {
        return a / b;
    }
}
