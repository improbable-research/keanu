package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class ReshapeVertex extends DoubleUnaryOpVertex {

    private int[] proposedShape;

    public ReshapeVertex(DoubleVertex inputVertex, int... proposedShape) {
        super(proposedShape, inputVertex);
        this.proposedShape = proposedShape;
    }

    /**
     * Returns the supplied vertex with a new shape of the same length
     */
    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.reshape(proposedShape);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex<?>, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).reshape(proposedShape);
    }

}
