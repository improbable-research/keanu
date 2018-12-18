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

@DisplayInformationForOutput(displayName = "*")
public class MultiplicationVertex extends DoubleBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param left  vertex to be multiplied
     * @param right vertex to be multiplied
     */
    @ExportVertexToPythonBindings
    public MultiplicationVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                                @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.times(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInput, PartialDerivative dRightWrtInput) {

        // dc = A * db + da * B;
        PartialDerivative partialsFromLeft = dLeftWrtInput.multiplyAlongOfDimensions(right.getValue(), left.getValue().getShape());
        PartialDerivative partialsFromRight = dRightWrtInput.multiplyAlongOfDimensions(left.getValue(), right.getValue().getShape());

        return partialsFromLeft.add(partialsFromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative dOutputsWrtLeft = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(right.getValue(), this.getShape());
        PartialDerivative dOutputsWrtRight = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(left.getValue(), this.getShape());

        partials.put(left, dOutputsWrtLeft);
        partials.put(right, dOutputsWrtRight);

        return partials;
    }
}
