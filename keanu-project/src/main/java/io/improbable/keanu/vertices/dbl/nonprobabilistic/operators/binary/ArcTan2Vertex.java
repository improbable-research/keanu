package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param left x coordinate
     * @param right y coordinate
     */
    public ArcTan2Vertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor left, DoubleTensor right) {
        return left.atan2(right);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber leftDual = dualNumbers.get(left);
        DualNumber rightDual = dualNumbers.get(right);

        DoubleTensor denominator = ((right.getValue().pow(2)).timesInPlace((left.getValue().pow(2))));

        PartialDerivatives thisInfLeft = leftDual.getPartialDerivatives().multiplyBy(right.getValue().div(denominator));
        PartialDerivatives thisInfRight = rightDual.getPartialDerivatives().multiplyBy((left.getValue().div(denominator)).unaryMinusInPlace());
        PartialDerivatives newInf = thisInfLeft.add(thisInfRight);
        return new DualNumber(op(left.getValue(), right.getValue()), newInf);
    }

    @Override
    protected Map<Vertex, PartialDerivatives> derivativeWithRespectTo(PartialDerivatives dAlldSelf) {
        return null;
    }
}
