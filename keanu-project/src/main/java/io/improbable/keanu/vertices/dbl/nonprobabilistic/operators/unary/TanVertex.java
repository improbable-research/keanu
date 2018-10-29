package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class TanVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the tangent of a vertex. Tan(vertex).
     *
     * @param inputVertex the vertex
     */
    public TanVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.tan();
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        DoubleTensor dTan = inputVertex.getValue().cos().powInPlace(2).reciprocalInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dTan, this.getValue().getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        //dTandInput = sec^2(x)
        DoubleTensor dTandInput = inputVertex.getValue().cos().powInPlace(2).reciprocalInPlace();

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dTandInput, this.getShape()));
        return partials;
    }
}
