package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DivisionVertex extends DoubleBinaryOpVertex {
    /**
     * Divides one vertex by another
     *
     * @param left  the vertex to be divided
     * @param right the vertex to divide
     */
    @ExportVertexToPythonBindings
    public DivisionVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.div(r);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dLeftWrtInputs, PartialDerivatives dRightWrtInputs) {

        // dc = (B * da - A * db) / B^2;
        PartialDerivatives partialsFromLeft;
        PartialDerivatives partialsFromRight;

        if (dLeftWrtInputs.isEmpty()) {
            partialsFromLeft = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromLeft = dLeftWrtInputs.multiplyAlongOfDimensions(right.getValue(), left.getValue().getShape());
        }

        if (dRightWrtInputs.isEmpty()) {
            partialsFromRight = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromRight = dRightWrtInputs.multiplyAlongOfDimensions(left.getValue(), right.getValue().getShape());
        }

        PartialDerivatives dSelfWrtInputs;
        if (partialsFromLeft.isEmpty() && partialsFromRight.isEmpty()) {
            dSelfWrtInputs = PartialDerivatives.OF_CONSTANT;
        } else {
            dSelfWrtInputs = partialsFromLeft.subtract(partialsFromRight).divideBy(right.getValue().pow(2));
        }

        return dSelfWrtInputs;
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();
        DoubleTensor dOutWrtLeft = rightValue.reciprocal();
        DoubleTensor dOutWrtRight = leftValue.div(rightValue.pow(2.0)).unaryMinusInPlace();
        partials.put(left, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(dOutWrtLeft, this.getShape()));
        partials.put(right, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(dOutWrtRight, this.getShape()));
        return partials;
    }
}
