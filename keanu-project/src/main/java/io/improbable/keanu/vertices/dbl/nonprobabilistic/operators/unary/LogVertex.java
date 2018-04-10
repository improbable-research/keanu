package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class LogVertex extends DoubleUnaryOpVertex {

    private final double logBase;

    public LogVertex(double base, DoubleVertex input) {
        super(input);
        this.logBase = Math.log(base);
    }

    public LogVertex(DoubleVertex input) {
        this(Math.E, input);
    }

    @Override
    public DualNumber getDualNumber() {

        DualNumber inputDualNumber = inputVertex.getDualNumber();
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().divideBy(inputDualNumber.getValue() * logBase);

        return new DualNumber(op(inputDualNumber.getValue()), outputInfinitesimal);
    }

    @Override
    protected Double op(Double value) {
        return Math.log(value) / logBase;
    }
}
