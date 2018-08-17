package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    /**
     * Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
     *
     * @param left  x coordinate
     * @param right y coordinate
     */
    public ArcTan2Vertex(DoubleVertex left, DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.atan2(r);
    }

    @Override
    protected DualNumber dualOp(DualNumber a, DualNumber b) {
        DoubleTensor denominator = ((b.getValue().pow(2)).plusInPlace((a.getValue().pow(2))));

        PartialDerivatives thisInfA = a.getPartialDerivatives().multiplyBy(b.getValue().div(denominator));
        PartialDerivatives thisInfB = b.getPartialDerivatives().multiplyBy((a.getValue().div(denominator)).unaryMinusInPlace());
        PartialDerivatives newInf = thisInfA.add(thisInfB);
        return new DualNumber(a.getValue().atan2(b.getValue()), newInf);
    }
}
