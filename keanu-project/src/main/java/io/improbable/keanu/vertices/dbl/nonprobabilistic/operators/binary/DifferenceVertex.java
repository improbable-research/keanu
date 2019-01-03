package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

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

        PartialDerivative fromLeft = AutoDiffBroadcast.correctForScalarPartialForward(dLeftWrtInput, left.getShape(), this.getShape());
        PartialDerivative fromRight = AutoDiffBroadcast.correctForScalarPartialForward(dRightWrtInput, right.getShape(), this.getShape());

        return fromLeft.subtract(fromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative toLeft = AutoDiffBroadcast.correctForScalarPartialReverse(derivativeOfOutputWithRespectToSelf, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForScalarPartialReverse(derivativeOfOutputWithRespectToSelf.multiplyBy(-1), this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);
        return partials;
    }
}
