package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class CosVertex extends DoubleUnaryOpVertex {

    public CosVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public CosVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.cos(a);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber inputDualNumber = inputVertex.getDualNumber();
        double dCos = -Math.sin(inputVertex.getValue());
        Infinitesimal outputInfinitesimal = inputDualNumber.getInfinitesimal().multiplyBy(dCos);
        return new DualNumber(Math.cos(inputVertex.getValue()), outputInfinitesimal);
    }
}
