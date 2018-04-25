package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

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
        return inputVertex.getDualNumber().sin();
    }
}
