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

public class ArcCosVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Takes the inverse cosine of a vertex, Arccos(vertex)
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public ArcCosVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.acos();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);

        DoubleTensor inputValue = inputVertex.getValue();

        DoubleTensor dArcCos = inputValue.unaryMinus().timesInPlace(inputValue).plusInPlace(1)
            .sqrtInPlace().reciprocalInPlace().unaryMinusInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dArcCos, inputVertex.getShape());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcCosdx = -1 / sqrt(1 - x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2).unaryMinusInPlace().plusInPlace(1)
            .sqrtInPlace()
            .reciprocalInPlace()
            .unaryMinusInPlace();

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtInput, this.getShape()));

        return partials;
    }
}