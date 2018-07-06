package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param a x coordinate
     * @param b y coordinate
     */
    public ArcTan2Vertex(DoubleVertex a, DoubleVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a, DoubleTensor b) {
        return a.atan2(b);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber aDual = dualNumbers.get(a);
        DualNumber bDual = dualNumbers.get(b);

        DoubleTensor denominator = ((b.getValue().pow(2)).timesInPlace((a.getValue().pow(2))));

        PartialDerivatives thisInfA = aDual.getPartialDerivatives().multiplyBy(b.getValue().div(denominator));
        PartialDerivatives thisInfB = bDual.getPartialDerivatives().multiplyBy((a.getValue().div(denominator)).unaryMinusInPlace());
        PartialDerivatives newInf = thisInfA.add(thisInfB);
        return new DualNumber(op(a.getValue(), b.getValue()), newInf);
    }
}
