package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ArcSinVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse sin of a vertex, Arcsin(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcSinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.asin();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.asin();
    }
}
