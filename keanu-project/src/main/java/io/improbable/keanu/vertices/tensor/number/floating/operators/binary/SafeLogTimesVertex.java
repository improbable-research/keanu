package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class SafeLogTimesVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    @ExportVertexToPythonBindings
    public SafeLogTimesVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> x,
                              @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> y) {
        super(x, y, x.ofType());
    }

    @Override
    protected TENSOR op(TENSOR x, TENSOR y) {
        return x.safeLogTimes(y);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        final ForwardModePartialDerivative dXWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        final ForwardModePartialDerivative dYWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        final ForwardModePartialDerivative fromLeft = AutoDiffBroadcast.correctForBroadcastPartialForward(dXWrtInput, left.getShape(), this.getShape());
        final ForwardModePartialDerivative fromRight = AutoDiffBroadcast.correctForBroadcastPartialForward(dYWrtInput, right.getShape(), this.getShape());

        final DoubleTensor x = left.getValue().toDouble();
        final DoubleTensor y = right.getValue().toDouble();
        final BooleanTensor yZeroMask = y.elementwiseEquals(0.0);

        final ForwardModePartialDerivative partialsFromX = fromLeft.multiply(y.div(x));
        final ForwardModePartialDerivative partialsFromY = fromRight.multiply(DoubleTensor.scalar(Double.NaN).where(yZeroMask, x.log()));

        return partialsFromX.add(partialsFromY);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        final Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();

        final DoubleTensor x = left.getValue().toDouble();
        final DoubleTensor y = right.getValue().toDouble();
        final BooleanTensor yZeroMask = y.elementwiseEquals(0.0);

        final ReverseModePartialDerivative dOutputsWrtX = derivativeOfOutputWithRespectToSelf.multiply(
            y.div(x)
        );

        final ReverseModePartialDerivative dOutputsWrtY = derivativeOfOutputWithRespectToSelf.multiply(
            DoubleTensor.scalar(Double.NaN).where(yZeroMask, x.log())
        );

        final ReverseModePartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtX, this.getShape(), left.getShape());
        final ReverseModePartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtY, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);

        return partials;
    }
}