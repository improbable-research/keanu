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

public class PowerVertex extends DoubleBinaryOpVertex {

    private static final String BASE_NAME = LEFT_NAME;
    private static final String EXPONENT_NAME = RIGHT_NAME;

    /**
     * Raises a vertex to the power of another
     *
     * @param base     the base vertex
     * @param exponent the exponent vertex
     */
    @ExportVertexToPythonBindings
    public PowerVertex(@LoadVertexParam(BASE_NAME) DoubleVertex base,
                       @LoadVertexParam(EXPONENT_NAME) DoubleVertex exponent) {
        super(base, exponent);
    }

    public DoubleVertex getBase() {
        return super.getLeft();
    }

    public DoubleVertex getExponent() {
        return super.getRight();
    }

    @Override
    protected DoubleTensor op(DoubleTensor base, DoubleTensor exponent) {
        return base.pow(exponent);
    }

    @Override
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dBaseWrtInput, PartialDerivative dExponentWrtInput) {

        PartialDerivative fromBase = AutoDiffBroadcast.correctForBroadcastPartialForward(dBaseWrtInput, left.getShape(), this.getShape());
        PartialDerivative fromExponent = AutoDiffBroadcast.correctForBroadcastPartialForward(dExponentWrtInput, right.getShape(), this.getShape());

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        PartialDerivative partialsFromBase;
        PartialDerivative partialsFromExponent;

        if (fromBase.isPresent()) {
            partialsFromBase = fromBase.multiplyAlongOfDimensions(
                right.getValue().times(left.getValue().pow(right.getValue().minus(1)))
            );
        } else {
            partialsFromBase = PartialDerivative.EMPTY;
        }

        if (fromExponent.isPresent()) {
            partialsFromExponent = fromExponent.multiplyAlongOfDimensions(
                left.getValue().log().timesInPlace(this.getValue())
            );
        } else {
            partialsFromExponent = PartialDerivative.EMPTY;
        }

        return partialsFromBase.add(partialsFromExponent);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor baseValue = getBase().getValue();
        DoubleTensor exponentValue = getExponent().getValue();
        DoubleTensor basePowExponent = getValue();
        DoubleTensor dSelfWrtBase = exponentValue.times(baseValue.pow(exponentValue.minus(1)));
        DoubleTensor dSelfWrtExponent = basePowExponent.times(baseValue.log());

        PartialDerivative dOutputsWrtBase = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtBase);
        PartialDerivative dOutputsWrtExponent = derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtExponent);

        PartialDerivative toBase = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtBase, this.getShape(), getBase().getShape());
        PartialDerivative toExponent = AutoDiffBroadcast.correctForBroadcastPartialReverse(dOutputsWrtExponent, this.getShape(), getExponent().getShape());

        partials.put(getBase(), toBase);
        partials.put(getExponent(), toExponent);
        return partials;
    }
}
