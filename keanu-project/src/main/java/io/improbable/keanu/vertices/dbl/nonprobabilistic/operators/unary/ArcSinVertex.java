package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.HashMap;
import java.util.Map;

public class ArcSinVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the inverse sin of a vertex, Arcsin(vertex)
     *
     * @param inputVertex the vertex
     */
    public ArcSinVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.asin();
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).asin();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcSindx = 1 / sqrt(1 - x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2).unaryMinusInPlace().plusInPlace(1)
            .sqrtInPlace()
            .reciprocalInPlace();

        Map<Vertex, PartialDerivatives> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyBy(dSelfWrtInput));

        return partials;
    }
}
