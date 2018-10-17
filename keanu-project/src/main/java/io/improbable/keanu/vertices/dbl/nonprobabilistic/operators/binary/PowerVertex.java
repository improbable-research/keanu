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
     * @param base  the base vertex
     * @param exponent the exponent vertex
     */
    public PowerVertex(DoubleVertex base, DoubleVertex exponent) {
        super(base, exponent);
    }

    @Override
    protected DoubleTensor op(DoubleTensor base, DoubleTensor exponent) {
        return base.pow(exponent);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives dBaseWrtInputs, PartialDerivatives dExponentWrtInputs) {

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        PartialDerivatives partialsFromBase;
        PartialDerivatives partialsFromExponent;

        if (dBaseWrtInputs.isEmpty()) {
            partialsFromBase = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromBase = dBaseWrtInputs.multiplyAlongOfDimensions(
                right.getValue().times(left.getValue().pow(right.getValue().minus(1))),
                this.getValue().getShape()
            );
        }

        if (dExponentWrtInputs.isEmpty()) {
            partialsFromExponent = PartialDerivatives.OF_CONSTANT;
        } else {
            partialsFromExponent = dExponentWrtInputs.multiplyAlongOfDimensions(
                left.getValue().log().timesInPlace(this.getValue()),
                right.getValue().getShape()
            );
        }

        return partialsFromBase.add(partialsFromExponent);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        DoubleTensor baseValue = getBase().getValue();
        DoubleTensor exponentValue = getExponent().getValue();
        DoubleTensor basePowExponent = getValue();
        DoubleTensor dSelfWrtBase = exponentValue.div(baseValue).timesInPlace(basePowExponent);
        DoubleTensor dSelfWrtExponent = basePowExponent.times(baseValue.log());
        partials.put(getBase(), derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtBase, this.getShape()));
        partials.put(getExponent(), derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtExponent, this.getShape()));
        return partials;
    }

    public DoubleVertex getBase() {
        return super.getLeft();
    }

    public DoubleVertex getExponent() {
        return super.getRight();
    }
}
