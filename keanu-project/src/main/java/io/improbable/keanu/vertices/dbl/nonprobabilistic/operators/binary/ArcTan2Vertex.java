package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    private static final String X_NAME = LEFT_NAME;
    private static final String Y_NAME = RIGHT_NAME;

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param x x coordinate
     * @param y y coordinate
     */
    @ExportVertexToPythonBindings
    public ArcTan2Vertex(@LoadVertexParam(X_NAME) DoubleVertex x,
                         @LoadVertexParam(Y_NAME) DoubleVertex y) {
        super(x, y);
    }

    @Override
    protected DoubleTensor op(DoubleTensor x, DoubleTensor y) {
        return x.atan2(y);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dxWrtInput, PartialDerivative dyWrtInput) {

        DoubleTensor yValue = right.getValue();
        DoubleTensor xValue = left.getValue();
        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));

        PartialDerivative fromX = AutoDiffBroadcast.correctForScalarPartialForward(dxWrtInput, left.getShape(), this.getShape());
        PartialDerivative fromY = AutoDiffBroadcast.correctForScalarPartialForward(dyWrtInput, right.getShape(), this.getShape());

        PartialDerivative diffFromX = fromX.multiplyAlongOfDimensions(yValue.div(denominator).unaryMinusInPlace());
        PartialDerivative diffFromY = fromY.multiplyAlongOfDimensions(xValue.div(denominator));

        return diffFromX.add(diffFromY);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor xValue = left.getValue();
        DoubleTensor yValue = right.getValue();

        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));
        DoubleTensor dOutWrtX = yValue.div(denominator).unaryMinusInPlace();
        DoubleTensor dOutWrtY = xValue.div(denominator);

        PartialDerivative dOutputsWrtLeft = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtX);
        PartialDerivative dOutputsWrtRight = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtY);

        PartialDerivative toLeft = AutoDiffBroadcast.correctForScalarPartialReverse(dOutputsWrtLeft, this.getShape(), left.getShape());
        PartialDerivative toRight = AutoDiffBroadcast.correctForScalarPartialReverse(dOutputsWrtRight, this.getShape(), right.getShape());

        partials.put(left, toLeft);
        partials.put(right, toRight);
        return partials;
    }
}
