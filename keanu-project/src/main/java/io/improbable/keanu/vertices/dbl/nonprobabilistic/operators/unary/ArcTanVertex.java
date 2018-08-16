package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class ArcTanVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse tan of a vertex, Arctan(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcTanVertex(DoubleVertex inputVertex) {
        super(inputVertex);;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.atan();
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.atan();
    }
}
