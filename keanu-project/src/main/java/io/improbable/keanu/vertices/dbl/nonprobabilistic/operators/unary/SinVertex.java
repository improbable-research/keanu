package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;

public class SinVertex extends DoubleUnaryOpVertex {

    public SinVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    public SinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected Double op(Double a) {
        return Math.sin(a);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber inputDualNumber = inputVertex.getDualNumber();
        double dSin = Math.cos(inputVertex.getValue());
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().multiplyBy(dSin);
        return new DualNumber(Math.sin(inputVertex.getValue()), outputInfinitesimal);
    }
}
