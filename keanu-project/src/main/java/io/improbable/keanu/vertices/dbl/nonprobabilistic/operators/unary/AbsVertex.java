package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;


import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class AbsVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the absolute of a vertex
     *
     * @param inputVertex the vertex
     */
    public AbsVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.abs();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        throw new UnsupportedOperationException();
    }
}
