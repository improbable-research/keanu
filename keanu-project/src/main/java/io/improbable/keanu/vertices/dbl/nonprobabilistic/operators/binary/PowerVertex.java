package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class PowerVertex extends DoubleBinaryOpVertex {

    /**
     * Raises a vertex to the power of another
     *
     * @param left  the base vertex
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
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives l, PartialDerivatives r) {

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        PartialDerivatives thisInfBase;
        PartialDerivatives thisInfExponent;

        if (l.isEmpty()) {
            thisInfBase = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfBase = l.multiplyAlongOfDimensions(
                right.getValue().times(left.getValue().pow(right.getValue().minus(1))),
                this.getValue().getShape()
            );
        }

        if (r.isEmpty()) {
            thisInfExponent = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfExponent = r.multiplyAlongOfDimensions(
                left.getValue().log().timesInPlace(this.getValue()),
                right.getValue().getShape()
            );
        }

        return thisInfBase.add(thisInfExponent);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor leftValue = left.getValue();
        DoubleTensor rightValue = right.getValue();
        DoubleTensor leftPowRight = getValue();
        DoubleTensor dOutWrtLeft = rightValue.div(leftValue).timesInPlace(leftPowRight);
        DoubleTensor dOutWrtRight = leftPowRight.times(leftValue.log());
        partials.put(left, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtLeft, this.getShape()));
        partials.put(right, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dOutWrtRight, this.getShape()));
        return partials;
    }

    public DoubleVertex getBase() {
        return super.getLeft();
    }

    public DoubleVertex getExponent() {
        return super.getRight();
    }
}
