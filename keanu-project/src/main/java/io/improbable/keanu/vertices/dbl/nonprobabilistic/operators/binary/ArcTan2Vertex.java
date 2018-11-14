package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    private static final String X_NAME = "x";
    private static final String Y_NAME = "y";

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param x x coordinate
     * @param y y coordinate
     */
    @ExportVertexToPythonBindings
    public ArcTan2Vertex(@LoadParentVertex(X_NAME) DoubleVertex x,
                         @LoadParentVertex(Y_NAME) DoubleVertex y) {
        super(x, y);
    }

    @SaveParentVertex(X_NAME)
    public DoubleVertex getX() {
        return super.getLeft();
    }

    @SaveParentVertex(Y_NAME)
    public DoubleVertex getY() {
        return super.getRight();
    }

    @Override
    protected DoubleTensor op(DoubleTensor x, DoubleTensor y) {
        return x.atan2(y);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dxWrtInputs, PartialDerivatives dyWrtInputs) {
        DoubleTensor yValue = right.getValue();
        DoubleTensor xValue = left.getValue();

        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));

        PartialDerivatives diffFromX = dxWrtInputs.multiplyAlongOfDimensions(
            yValue.div(denominator).unaryMinusInPlace(),
            xValue.getShape()
        );

        PartialDerivatives diffFromY = dyWrtInputs.multiplyAlongOfDimensions(
            xValue.div(denominator),
            yValue.getShape()
        );

        return diffFromX.add(diffFromY);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor xValue = left.getValue();
        DoubleTensor yValue = right.getValue();

        DoubleTensor denominator = yValue.pow(2).plusInPlace(xValue.pow(2));
        DoubleTensor dOutWrtX = yValue.div(denominator).unaryMinusInPlace();
        DoubleTensor dOutWrtY = xValue.div(denominator);

        partials.put(left, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtX, this.getShape()));
        partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtY, this.getShape()));
        return partials;
    }
}
