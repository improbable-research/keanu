package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary.BinaryOpVertex.correctForScalarPartialForward;

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

        PartialDerivative fromBase = correctForScalarPartialForward(dBaseWrtInput, this.getShape(), left.getShape());
        PartialDerivative fromExponent = correctForScalarPartialForward(dExponentWrtInput, this.getShape(), right.getShape());

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        PartialDerivative partialsFromBase;
        PartialDerivative partialsFromExponent;

        if (fromBase.isPresent()) {
            partialsFromBase = fromBase.multiplyAlongOfDimensions(
                right.getValue().times(left.getValue().pow(right.getValue().minus(1))),
                this.getValue().getShape()
            );
        } else {
            partialsFromBase = PartialDerivative.EMPTY;
        }

        if (fromExponent.isPresent()) {
            partialsFromExponent = fromExponent.multiplyAlongOfDimensions(
                left.getValue().log().timesInPlace(this.getValue()),
                right.getValue().getShape()
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
        DoubleTensor dSelfWrtBase = exponentValue.div(baseValue).timesInPlace(basePowExponent);
        DoubleTensor dSelfWrtExponent = basePowExponent.times(baseValue.log());
        partials.put(getBase(), derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtBase, this.getShape()));
        partials.put(getExponent(), derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtExponent, this.getShape()));
        return partials;
    }
}
