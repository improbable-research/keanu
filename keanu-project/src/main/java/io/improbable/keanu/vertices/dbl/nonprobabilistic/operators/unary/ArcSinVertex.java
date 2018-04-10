package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;

public class ArcSinVertex extends DoubleUnaryOpVertex {

    public ArcSinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public ArcSinVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.asin(a);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber inputDualNumber = inputVertex.getDualNumber();
        double dArcSin = 1 / Math.sqrt(1 - Math.pow(inputVertex.getValue(), 2));
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().multiplyBy(dArcSin);
        return new DualNumber(Math.asin(inputVertex.getValue()), outputInfinitesimal);
    }
}
