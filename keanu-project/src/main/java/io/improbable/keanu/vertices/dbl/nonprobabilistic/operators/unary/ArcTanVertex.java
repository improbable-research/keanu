package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ArcTanVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse tan of a vertex, Arctan(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcTanVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.atan();
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        DoubleTensor value = inputVertex.getValue();

        DoubleTensor dArcTan = value.pow(2).plusInPlace(1).reciprocalInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dArcTan, inputVertex.getShape());
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcTandx = 1 / (1 + x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2).plusInPlace(1).reciprocalInPlace();

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtInput, this.getShape()));

        return partials;
    }
}
