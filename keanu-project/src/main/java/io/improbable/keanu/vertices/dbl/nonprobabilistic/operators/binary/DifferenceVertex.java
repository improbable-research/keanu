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
import static io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpVertex.correctForScalarPartial;
import static io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpVertex.shouldCorrectPartialForScalar;


@DisplayInformationForOutput(displayName = "-")
public class DifferenceVertex extends DoubleBinaryOpVertex {

    /**
     * Subtracts one vertex from another
     *
     * @param left  the vertex that will be subtracted from
     * @param right the vertex to subtract
     */
    @ExportVertexToPythonBindings
    public DifferenceVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                            @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.minus(r);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dLeftWrtInput, PartialDerivative dRightWrtInput) {

        boolean shouldCorrectForLeftScalar = shouldCorrectPartialForScalar(dLeftWrtInput, this.getShape(), left.getShape());
        PartialDerivative fromLeft = shouldCorrectForLeftScalar ? correctForScalarPartial(dLeftWrtInput, this.getShape(), left.getShape().length) : dLeftWrtInput;
        boolean shouldCorrectForRightScalar = shouldCorrectPartialForScalar(dRightWrtInput, this.getShape(), right.getShape());
        PartialDerivative fromRight = shouldCorrectForRightScalar ? correctForScalarPartial(dRightWrtInput, this.getShape(), right.getShape().length) : dRightWrtInput;

        return fromLeft.subtract(fromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputWithRespectToSelf);
        partials.put(right, derivativeOfOutputWithRespectToSelf.multiplyBy(-1.0));
        return partials;
    }
}
