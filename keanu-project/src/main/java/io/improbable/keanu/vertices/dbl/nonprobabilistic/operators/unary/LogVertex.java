package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;

public class LogVertex extends DoubleUnaryOpVertex {


    public LogVertex(DoubleVertex input) {
        super(input);
    }

    @Override
    protected Double op(Double value) {
        return Math.log(value);
    }

    @Override
    public DualNumber getDualNumber() {
        return inputVertex.getDualNumber().log();
    }

}
