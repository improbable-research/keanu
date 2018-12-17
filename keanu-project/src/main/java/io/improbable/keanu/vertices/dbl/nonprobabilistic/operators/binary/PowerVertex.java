package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class PowerVertex extends DoubleBinaryOpVertex {

    private static final String BASE_NAME = LEFT_NAME;
    private static final String EXPONENT_NAME = RIGHT_NAME;

    /**
     * Raises a vertex to the power of another
     *
     * @param base  the base vertex
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
    protected PartialDerivative forwardModeAutoDifferentiation(PartialDerivative dBaseWrtInputs, PartialDerivative dExponentWrtInputs) {

        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        PartialDerivative partialsFromBase;
        PartialDerivative partialsFromExponent;

        if (dBaseWrtInputs.isPresent()) {
            partialsFromBase = PartialDerivative.OF_CONSTANT;
        } else {
            partialsFromBase = dBaseWrtInputs.multiplyAlongOfDimensions(
                right.getValue().times(left.getValue().pow(right.getValue().minus(1))),
                this.getValue().getShape()
            );
        }

        if (dExponentWrtInputs.isPresent()) {
            partialsFromExponent = PartialDerivative.OF_CONSTANT;
        } else {
            partialsFromExponent = dExponentWrtInputs.multiplyAlongOfDimensions(
                left.getValue().log().timesInPlace(this.getValue()),
                right.getValue().getShape()
            );
        }

        return partialsFromBase.add(partialsFromExponent);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        DoubleTensor baseValue = getBase().getValue();
        DoubleTensor exponentValue = getExponent().getValue();
        DoubleTensor basePowExponent = getValue();
        DoubleTensor dSelfWrtBase = exponentValue.div(baseValue).timesInPlace(basePowExponent);
        DoubleTensor dSelfWrtExponent = basePowExponent.times(baseValue.log());
        partials.put(getBase(), derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtBase, this.getShape()));
        partials.put(getExponent(), derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtExponent, this.getShape()));
        return partials;
    }
}
