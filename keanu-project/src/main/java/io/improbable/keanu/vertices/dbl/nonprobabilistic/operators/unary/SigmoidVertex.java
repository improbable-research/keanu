package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class SigmoidVertex extends DoubleUnaryOpVertex {

    public SigmoidVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public SigmoidVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double x) {
        return 1. / (1. + Math.exp(-x));
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber dualNumber = inputVertex.getDualNumber();
        double x = dualNumber.getValue();
        double dxdfx = Math.exp(x) / Math.pow(Math.exp(x) + 1., 2);
        Infinitesimal infinitesimal = dualNumber.getInfinitesimal().multiplyBy(dxdfx);
        return new DualNumber(op(x), infinitesimal);
    }
}
