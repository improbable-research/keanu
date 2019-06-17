package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ArcSinVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Takes the inverse sin of a vertex, Arcsin(vertex)
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public ArcSinVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.asin();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        DoubleTensor inputValue = inputVertex.getValue();

        DoubleTensor dArcSin = (inputValue.unaryMinus().timesInPlace(inputValue).plusInPlace(1.0))
            .sqrtInPlace().reciprocalInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dArcSin);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcSindx = 1 / sqrt(1 - x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2.0).unaryMinusInPlace().plusInPlace(1.0)
            .sqrtInPlace()
            .reciprocalInPlace();

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtInput));

        return partials;
    }
}
