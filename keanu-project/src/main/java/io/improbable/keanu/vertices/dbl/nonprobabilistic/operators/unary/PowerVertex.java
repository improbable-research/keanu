package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;

public class PowerVertex extends DoubleBinaryOpVertex {

    public PowerVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    public PowerVertex(DoubleVertex a, double b) {
        super(a, new ConstantDoubleVertex(b));
    }

    @Override
    protected Double op(Double a, Double b) {
        return Math.pow(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber inputDualNumber = a.getDualNumber();
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().multiplyBy(b.getValue() * (Math.pow(a.getValue(), b.getValue() - 1.0)));
        return new DualNumber(op(inputDualNumber.getValue(), b.getDualNumber().getValue()), outputInfinitesimal);
    }
}
