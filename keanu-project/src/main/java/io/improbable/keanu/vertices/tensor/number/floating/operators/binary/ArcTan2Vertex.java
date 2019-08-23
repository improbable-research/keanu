package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
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
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ArcTan2Vertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String X_NAME = LEFT_NAME;
    private static final String Y_NAME = RIGHT_NAME;

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param x x coordinate
     * @param y y coordinate
     */
    @ExportVertexToPythonBindings
    public ArcTan2Vertex(@LoadVertexParam(X_NAME) TensorVertex<T, TENSOR, VERTEX> x,
                         @LoadVertexParam(Y_NAME) TensorVertex<T, TENSOR, VERTEX> y) {
        super(x, y, x.ofType());
    }

    @Override
    protected TENSOR op(TENSOR x, TENSOR y) {
        return x.atan2(y);
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dxWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        ForwardModePartialDerivative dyWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        DoubleTensor yValue = right.getValue().toDouble();
        DoubleTensor xValue = left.getValue().toDouble();
        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));

        ForwardModePartialDerivative fromX = AutoDiffBroadcast.correctForBroadcastPartialForward(dxWrtInput, left.getShape(), this.getShape());
        ForwardModePartialDerivative fromY = AutoDiffBroadcast.correctForBroadcastPartialForward(dyWrtInput, right.getShape(), this.getShape());

        ForwardModePartialDerivative diffFromX = fromX.multiply(yValue.div(denominator).unaryMinusInPlace());
        ForwardModePartialDerivative diffFromY = fromY.multiply(xValue.div(denominator));

        return diffFromX.add(diffFromY);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor xValue = left.getValue().toDouble();
        DoubleTensor yValue = right.getValue().toDouble();

        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));
        DoubleTensor dOutWrtX = yValue.div(denominator).unaryMinusInPlace();
        DoubleTensor dOutWrtY = xValue.div(denominator);

        PartialDerivative dOutputsWrtLeft = derivativeOfOutputWithRespectToSelf.multiply(dOutWrtX);
        PartialDerivative dOutputsWrtRight = derivativeOfOutputWithRespectToSelf.multiply(dOutWrtY);

        PartialDerivative toLeft = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtLeft, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtRight, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);
        return partials;
    }
}
