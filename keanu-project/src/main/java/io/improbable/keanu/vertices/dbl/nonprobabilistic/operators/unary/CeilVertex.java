package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class CeilVertex extends DoubleUnaryOpVertex {

    public CeilVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public CeilVertex(double inputValue) {
        this(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.ceil(a);
    }

    @Override
    public DualNumber getDualNumber() {
        throw new UnsupportedOperationException();
    }
}
