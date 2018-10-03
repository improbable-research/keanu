package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.HashMap;
import java.util.Map;

public class PowerVertex extends DoubleBinaryOpVertex {

    /**
     * Raises a vertex to the power of another
     *
     * @param left the base vertex
     * @param right the exponent vertex
     */
    public PowerVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right);
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.pow(r);
    }

    @Override
    protected DualNumber dualOp(DualNumber l, DualNumber r) {
        return l.pow(r);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(
            PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();
        DoubleTensor leftPowRight = getValue();
        DoubleTensor dOutWrtLeft = rightValue.div(leftValue).timesInPlace(leftPowRight);
        DoubleTensor dOutWrtRight = leftPowRight.times(leftValue.log());
        partials.put(
                left,
                derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(
                        dOutWrtLeft, this.getShape()));
        partials.put(
                right,
                derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(
                        dOutWrtRight, this.getShape()));
        return partials;
    }

    public DoubleVertex getBase() {
        return super.getLeft();
    }

    public DoubleVertex getExponent() {
        return super.getRight();
    }
}
