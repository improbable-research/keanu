package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class PowerVertex extends DoubleBinaryOpVertex {

    /**
     * Raises a vertex to the power of another
     *
     * @param left the base vertex
     * @param right the exponent vertex
     */
    public PowerVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor left, DoubleTensor right) {
        return left.pow(right);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber leftDual = dualNumbers.get(left);
        DualNumber rightDual = dualNumbers.get(right);
        return leftDual.pow(rightDual);
    }

    @Override
    protected Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();
        DoubleTensor leftPowRight = getValue();
        DoubleTensor dOutWrtLeft = rightValue.div(leftValue).timesInPlace(leftPowRight);
        DoubleTensor dOutWrtRight = leftPowRight.times(leftValue.log());
        partials.put(left, Differentiator.reshapeReverseAutoDiff(derivativeOfOutputsWithRespectToSelf.multiplyBy(dOutWrtLeft), left.getValue(), right.getValue()));
        partials.put(right, Differentiator.reshapeReverseAutoDiff(derivativeOfOutputsWithRespectToSelf.multiplyBy(dOutWrtRight), right.getValue(), left.getValue()));
        return partials;
    }

    public DoubleVertex getBase(){
        return super.getLeft();
    }

    public DoubleVertex getExponent(){
        return super.getRight();
    }
}
