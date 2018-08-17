package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class RoundVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the Rounding operator to a vertex.
     * This maps a vertex to the nearest integer value
     *
     * @param inputVertex the vertex to be rounded
     */
    public RoundVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.round();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        throw new UnsupportedOperationException();
    }
}
