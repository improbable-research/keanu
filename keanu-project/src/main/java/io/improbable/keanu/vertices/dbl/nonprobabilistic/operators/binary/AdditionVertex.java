package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
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
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInputs, PartialDerivative dRightWrtInputs) {
        return dLeftWrtInputs.add(dRightWrtInputs,
            TensorShape.isLengthOne(left.getShape()),
            TensorShape.isLengthOne(right.getShape()),
            this.getShape()
        );
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputsWithRespectToSelf);
        partials.put(right, derivativeOfOutputsWithRespectToSelf);
        return partials;
    }

}
