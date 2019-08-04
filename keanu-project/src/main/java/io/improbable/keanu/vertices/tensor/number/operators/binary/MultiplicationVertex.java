package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

@DisplayInformationForOutput(displayName = "*")
public class MultiplicationVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * Multiplies one vertex by another
     *
     * @param left  vertex to be multiplied
     * @param right vertex to be multiplied
     */
    @ExportVertexToPythonBindings
    public MultiplicationVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(left, right, left.ofType());
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.times(r);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
        PartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);

        PartialDerivative fromLeft = AutoDiffBroadcast.correctForBroadcastPartialForward(dLeftWrtInput, left.getShape(), this.getShape());
        PartialDerivative fromRight = AutoDiffBroadcast.correctForBroadcastPartialForward(dRightWrtInput, right.getShape(), this.getShape());

        // dc = A * db + da * B;
        PartialDerivative partialsFromLeft = fromLeft.multiplyAlongOfDimensions(right.getValue().toDouble(), this.getRank());
        PartialDerivative partialsFromRight = fromRight.multiplyAlongOfDimensions(left.getValue().toDouble(), this.getRank());

        return partialsFromLeft.add(partialsFromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative dOutputsWrtLeft = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            right.getValue().toDouble()
        );

        PartialDerivative dOutputsWrtRight = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            left.getValue().toDouble()
        );

        PartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtLeft, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtRight, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);

        return partials;
    }
}
