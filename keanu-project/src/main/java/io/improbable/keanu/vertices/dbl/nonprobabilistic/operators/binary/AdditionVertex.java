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

@DisplayInformationForOutput(displayName = "+")
public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left  a vertex to add
     * @param right a vertex to add
     */
    @ExportVertexToPythonBindings
    public AdditionVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                          @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.plus(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInput, PartialDerivative dRightWrtInput) {

        PartialDerivative fromLeft = correctForScalarPartialForward(dLeftWrtInput, this.getShape(), left.getShape());
        PartialDerivative fromRight = correctForScalarPartialForward(dRightWrtInput, this.getShape(), right.getShape());

        return fromLeft.add(fromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative toLeft = correctForScalarReverse(derivativeOfOutputWithRespectToSelf, this.getShape(), left.getShape());
        PartialDerivative toRight = correctForScalarReverse(derivativeOfOutputWithRespectToSelf, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);
        return partials;
    }

}
