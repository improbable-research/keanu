package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
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

        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();
        DoubleTensor denominator = rightValue.pow(2).plusInPlace(leftValue.pow(2));

        PartialDerivatives thisInfLeft = leftDual.getPartialDerivatives().multiplyBy(rightValue.div(denominator));
        PartialDerivatives thisInfRight = rightDual.getPartialDerivatives().multiplyBy(leftValue.div(denominator).unaryMinusInPlace());
        PartialDerivatives newInf = thisInfLeft.add(thisInfRight);
        return new DualNumber(op(leftValue, rightValue), newInf);
    }

    @Override
    protected Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();

        DoubleTensor denominator = rightValue.pow(2).plusInPlace(leftValue.pow(2));
        DoubleTensor dOutWrtLeft = rightValue.divInPlace(denominator);
        DoubleTensor dOutWrtRight = leftValue.div(denominator).unaryMinusInPlace();

        partials.put(left, derivativeOfOutputsWithRespectToSelf.multiplyBy(dOutWrtLeft));
        partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyBy(dOutWrtRight));
        return partials;
    }
}
