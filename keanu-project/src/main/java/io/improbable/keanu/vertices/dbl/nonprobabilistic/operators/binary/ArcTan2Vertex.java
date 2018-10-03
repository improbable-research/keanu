package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param x x coordinate
     * @param y y coordinate
     */
    public ArcTan2Vertex(DoubleVertex x, DoubleVertex y) {
        super(x, y);
    }

    @Override
    protected DoubleTensor op(DoubleTensor x, DoubleTensor y) {
        return x.atan2(y);
    }

    @Override
    protected PartialDerivatives dualOp(PartialDerivatives x, PartialDerivatives y) {
        DoubleTensor yValue = right.getValue();
        DoubleTensor xValue = left.getValue();

        DoubleTensor denominator = ((yValue.pow(2)).plusInPlace((xValue.pow(2))));

        PartialDerivatives thisInfX = x
            .multiplyAlongOfDimensions((yValue.div(denominator)).unaryMinusInPlace(), xValue.getShape());
        PartialDerivatives thisInfY = y
            .multiplyAlongOfDimensions(xValue.div(denominator), yValue.getShape());

        return thisInfX.add(thisInfY);
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
