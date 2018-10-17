package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class SinVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the sine of a vertex. Sin(vertex).
     *
     * @param inputVertex the vertex
     */
    public SinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.sin();
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        DoubleTensor dSin = inputVertex.getValue().cos();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dSin, this.getValue().getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf
            .multiplyAlongWrtDimensions(inputVertex.getValue().cos(), this.getShape()));
        return partials;
    }
}
