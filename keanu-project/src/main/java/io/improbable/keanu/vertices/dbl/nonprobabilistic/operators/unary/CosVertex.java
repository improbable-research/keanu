package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class CosVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the cosine of a vertex, Cos(vertex)
     *
     * @param inputVertex the vertex
     */
    public CosVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.cos();
    }

    @Override
    protected PartialDerivatives dualOp(PartialDerivatives partialDerivatives) {

        DoubleTensor inputValue = inputVertex.getValue();

        if (partialDerivatives.isEmpty()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            DoubleTensor dCos = inputValue.sin().unaryMinusInPlace();
            return partialDerivatives.multiplyAlongOfDimensions(dCos, inputValue.getShape());
        }
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(inputVertex.getValue().sin().unaryMinusInPlace(), this.getShape()));
        return partials;
    }
}
