package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ExpVertex extends DoubleUnaryOpVertex {

    /**
     * Calculates the exponential of an input vertex
     *
     * @param inputVertex the vertex
     */
    public ExpVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.exp();
    }

    @Override
    protected PartialDerivatives dualOp(PartialDerivatives partialDerivatives) {

        if (partialDerivatives.isEmpty()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            return partialDerivatives.multiplyAlongOfDimensions(this.getValue(), inputVertex.getShape());
        }
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(getValue(), this.getShape()));
        return partials;
    }
}
