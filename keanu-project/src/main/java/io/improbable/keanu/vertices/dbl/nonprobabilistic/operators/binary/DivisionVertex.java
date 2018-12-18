package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

@DisplayInformationForOutput(displayName = "/")
public class DivisionVertex extends DoubleBinaryOpVertex {
    /**
     * Divides one vertex by another
     *
     * @param left  the vertex to be divided
     * @param right the vertex to divide
     */
    @ExportVertexToPythonBindings
    public DivisionVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                          @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.div(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInputs, PartialDerivative dRightWrtInputs) {

        // dc = (B * da - A * db) / B^2;
        PartialDerivative partialsFromLeft;
        PartialDerivative partialsFromRight;

        if (dLeftWrtInputs.isPresent()) {
            partialsFromLeft = dLeftWrtInputs.multiplyAlongOfDimensions(right.getValue(), left.getValue().getShape());
        } else {
            partialsFromLeft = PartialDerivative.EMPTY;
        }

        if (dRightWrtInputs.isPresent()) {
            partialsFromRight = dRightWrtInputs.multiplyAlongOfDimensions(left.getValue(), right.getValue().getShape());
        } else {
            partialsFromRight = PartialDerivative.EMPTY;
        }

        PartialDerivative dSelfWrtInputs;
        if (partialsFromLeft.isEmpty() && partialsFromRight.isEmpty()) {
            dSelfWrtInputs = PartialDerivative.EMPTY;
        } else {
            dSelfWrtInputs = partialsFromLeft.subtract(partialsFromRight).divideBy(right.getValue().pow(2));
        }

        return dSelfWrtInputs;
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
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
