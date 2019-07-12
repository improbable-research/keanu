package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

@DisplayInformationForOutput(displayName = "*")
public class MultiplicationVertex extends DoubleBinaryOpVertex implements Differentiable {

    /**
     * Multiplies one vertex by another
     *
     * @param left  vertex to be multiplied
     * @param right vertex to be multiplied
     */
    @ExportVertexToPythonBindings
    public MultiplicationVertex(@LoadVertexParam(LEFT_NAME) Vertex<DoubleTensor, ?> left,
                                @LoadVertexParam(RIGHT_NAME) Vertex<DoubleTensor, ?> right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.times(r);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
        PartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);

        PartialDerivative fromLeft = AutoDiffBroadcast.correctForBroadcastPartialForward(dLeftWrtInput, left.getShape(), this.getShape());
        PartialDerivative fromRight = AutoDiffBroadcast.correctForBroadcastPartialForward(dRightWrtInput, right.getShape(), this.getShape());

        // dc = A * db + da * B;
        PartialDerivative partialsFromLeft = fromLeft.multiplyAlongOfDimensions(right.getValue(), this.getRank());
        PartialDerivative partialsFromRight = fromRight.multiplyAlongOfDimensions(left.getValue(), this.getRank());

        return partialsFromLeft.add(partialsFromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative dOutputsWrtLeft = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            right.getValue()
        );

        PartialDerivative dOutputsWrtRight = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            left.getValue()
        );

        PartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtLeft, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtRight, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);

        return partials;
    }
}
