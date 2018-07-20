package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left a vertex to add
     * @param right a vertex to add
     */
    public AdditionVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber leftDual = dualNumbers.get(left);
        DualNumber rightDual = dualNumbers.get(right);
        return leftDual.plus(rightDual);
    }

    @Override
    protected DoubleTensor op(DoubleTensor left, DoubleTensor right) {
        return left.plus(right);
    }
}
