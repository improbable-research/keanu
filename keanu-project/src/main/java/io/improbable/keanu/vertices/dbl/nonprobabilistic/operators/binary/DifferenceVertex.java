package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;


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

        boolean shouldCorrectForLeftScalar = shouldCorrectPartialForScalar(dLeftWrtInput, left.getShape());

        PartialDerivative fromLeft = shouldCorrectForLeftScalar ? correctForScalarPartial(dLeftWrtInput) : dLeftWrtInput;

        boolean shouldCorrectForRightScalar = shouldCorrectPartialForScalar(dRightWrtInput, right.getShape());

        PartialDerivative fromRight = shouldCorrectForRightScalar ? correctForScalarPartial(dRightWrtInput) : dRightWrtInput;

        return fromLeft.subtract(fromRight);
    }

    private boolean shouldCorrectPartialForScalar(PartialDerivative dSideWrtInput, long[] sideShape) {
        return dSideWrtInput.isPresent() && !Arrays.equals(sideShape, this.getShape());
    }

    private PartialDerivative correctForScalarPartial(PartialDerivative partialDerivative) {
        DoubleTensor partial = partialDerivative.getPartial();
        DoubleTensor correctedPartial = DoubleTensor.zeros(TensorShape.concat(this.getShape(), partial.getShape())).plus(partial);
        return new PartialDerivative(partialDerivative.getKey(), correctedPartial);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(left, derivativeOfOutputWithRespectToSelf);
        partials.put(right, derivativeOfOutputWithRespectToSelf.multiplyBy(-1.0));
        return partials;
    }
}
