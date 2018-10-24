package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class LogGammaVertex extends DoubleUnaryOpVertex {

    /**
     * Returns the log of the gamma of the inputVertex
     *
     * @param inputVertex the vertex
     */
    public LogGammaVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.logGamma();
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(inputVertex.getValue().digamma(), getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        PartialDerivatives dOutputsWrtInputVertex =
            derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(inputVertex.getValue().digamma(), getShape());
        partials.put(inputVertex, dOutputsWrtInputVertex);
        return partials;
    }
}
