package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class LogVertex extends DoubleUnaryOpVertex {

    /**
     * Returns the natural logarithm, base e, of a vertex
     *
     * @param inputVertex the vertex
     */
    public LogVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.log();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.log();
    }
}
