package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class PowerVertex extends DoubleBinaryOpVertex {

    public PowerVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    public PowerVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    public PowerVertex(double a, DoubleVertex b) {
        this(new ConstantDoubleVertex(a), b);
    }

    @Override
    protected Double op(Double a, Double b) {
        return Math.pow(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber aDual = a.getDualNumber();
        DualNumber bDual = b.getDualNumber();
        return aDual.pow(bDual);
    }
}
