package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ArcSinVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse sin of a vertex, Arcsin(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcSinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.asin();
    }

    @Override
    protected PartialDerivatives dualOp(PartialDerivatives partialDerivatives) {

        DoubleTensor inputValue = inputVertex.getValue();

        if (partialDerivatives.isEmpty()) {
            return PartialDerivatives.OF_CONSTANT;
        } else {
            DoubleTensor dArcSin = (inputValue.unaryMinus().timesInPlace(inputValue).plusInPlace(1))
                .sqrtInPlace().reciprocalInPlace();
            return partialDerivatives.multiplyAlongOfDimensions(dArcSin, inputValue.getShape());
        }
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcSindx = 1 / sqrt(1 - x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2).unaryMinusInPlace().plusInPlace(1)
            .sqrtInPlace()
            .reciprocalInPlace();

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtInput, this.getShape()));

        return partials;
    }
}
