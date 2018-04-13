package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class PowerVertex extends DoubleBinaryOpVertex {

    public PowerVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    public PowerVertex(DoubleVertex a, double b) {
        this(a, new ConstantDoubleVertex(b));
    }

    @Override
    protected Double op(Double a, Double b) {
        return Math.pow(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber aDual = a.getDualNumber();
        DualNumber bDual = b.getDualNumber();
        Infinitesimal outputInfinitesimal = aDual.getInfinitesimal().multiplyBy(b.getValue() * (Math.pow(a.getValue(), b.getValue() - 1.0)));
        return new DualNumber(op(aDual.getValue(), bDual.getValue()), outputInfinitesimal);
    }
}
