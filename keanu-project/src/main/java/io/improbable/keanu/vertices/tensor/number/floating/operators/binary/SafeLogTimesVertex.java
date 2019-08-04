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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

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
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        final PartialDerivative dXWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
        final PartialDerivative dYWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);

        final PartialDerivative fromLeft = AutoDiffBroadcast.correctForBroadcastPartialForward(dXWrtInput, left.getShape(), this.getShape());
        final PartialDerivative fromRight = AutoDiffBroadcast.correctForBroadcastPartialForward(dYWrtInput, right.getShape(), this.getShape());

        final DoubleTensor x = left.getValue().toDouble();
        final DoubleTensor y = right.getValue().toDouble();
        final BooleanTensor yZeroMask = y.elementwiseEquals(0.0);

        final PartialDerivative partialsFromX = fromLeft.multiplyAlongOfDimensions(y.div(x), this.getRank());
        final PartialDerivative partialsFromY = fromRight.multiplyAlongOfDimensions(DoubleTensor.scalar(Double.NaN).where(yZeroMask, x.log()), this.getRank());

        return partialsFromX.add(partialsFromY);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        final Map<Vertex, PartialDerivative> partials = new HashMap<>();

        final DoubleTensor x = left.getValue().toDouble();
        final DoubleTensor y = right.getValue().toDouble();
        final BooleanTensor yZeroMask = y.elementwiseEquals(0.0);

        final PartialDerivative dOutputsWrtX = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            y.div(x)
        );

        final PartialDerivative dOutputsWrtY = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(
            DoubleTensor.scalar(Double.NaN).where(yZeroMask, x.log())
        );

        final PartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtX, this.getShape(), left.getShape());
        final PartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtY, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);

        return partials;
    }
}