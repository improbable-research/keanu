package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

import java.util.Map;

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
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber dualNumber = dualNumbers.get(inputVertex);
        double x = dualNumber.getValue();
        double dxdfx = Math.exp(x) / Math.pow(Math.exp(x) + 1., 2);
        Infinitesimal infinitesimal = dualNumber.getInfinitesimal().multiplyBy(dxdfx);
        return new DualNumber(op(x), infinitesimal);
    }
}
