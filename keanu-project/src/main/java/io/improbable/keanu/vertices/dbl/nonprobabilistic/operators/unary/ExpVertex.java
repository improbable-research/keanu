package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ExpVertex extends DoubleUnaryOpVertex {

    /**
     * Calculates the exponential of an input vertex
     *
     * @param inputVertex the vertex
     */
    public ExpVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.exp();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.exp();
    }
}
