package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class PowerVertex extends DoubleUnaryOpVertex {

    private final double power;

    public PowerVertex(DoubleVertex input, double power) {
        super(input);
        this.power = power;
    }

    @Override
    protected Double op(Double a) {
        return Math.pow(a, power);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber inputDualNumber = inputVertex.getDualNumber();
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().multiplyBy(power * Math.pow(inputVertex.getValue(), power - 1.0));
        return new DualNumber(op(inputDualNumber.getValue()), outputInfinitesimal);
    }
}
