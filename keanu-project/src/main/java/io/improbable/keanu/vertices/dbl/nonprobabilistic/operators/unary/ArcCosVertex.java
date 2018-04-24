package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class ArcCosVertex extends DoubleUnaryOpVertex {

    public ArcCosVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public ArcCosVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.acos(a);
    }

    @Override
    public DualNumber getDualNumber() {
        return inputVertex.getDualNumber().acos();
    }
}