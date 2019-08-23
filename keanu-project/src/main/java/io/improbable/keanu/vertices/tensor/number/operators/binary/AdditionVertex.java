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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

@DisplayInformationForOutput(displayName = "+")
public class AdditionVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * Adds one vertex to another
     *
     * @param left  a vertex to add
     * @param right a vertex to add
     */
    @ExportVertexToPythonBindings
    public AdditionVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                          @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(left, right, left.ofType());
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.plus(r);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        ForwardModePartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        ForwardModePartialDerivative fromLeft = AutoDiffBroadcast.correctForBroadcastPartialForward(dLeftWrtInput, left.getShape(), this.getShape());
        ForwardModePartialDerivative fromRight = AutoDiffBroadcast.correctForBroadcastPartialForward(dRightWrtInput, right.getShape(), this.getShape());

        return fromLeft.add(fromRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        PartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(derivativeOfOutputWithRespectToSelf, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(derivativeOfOutputWithRespectToSelf, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);
        return partials;
    }
}
