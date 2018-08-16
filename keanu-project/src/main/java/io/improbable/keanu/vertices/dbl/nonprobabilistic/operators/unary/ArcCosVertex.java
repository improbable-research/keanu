package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ArcCosVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse cosine of a vertex, Arccos(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcCosVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.acos();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.acos();
    }
}