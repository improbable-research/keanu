package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class ExpVertex extends DoubleUnaryOpVertex {

    public ExpVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public ExpVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.exp(a);
    }

    @Override
    public DualNumber getDualNumber() {
        return inputVertex.getDualNumber().exp();
    }

}
