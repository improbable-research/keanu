package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

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
        return inputVertex.getDualNumber().asin();
    }
}
