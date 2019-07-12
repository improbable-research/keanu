package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ArcTanVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Takes the inverse tan of a vertex, Arctan(vertex)
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public ArcTanVertex(@LoadVertexParam(INPUT_VERTEX_NAME) Vertex<DoubleTensor, ?> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.atan();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        DoubleTensor value = inputVertex.getValue();

        DoubleTensor dArcTan = value.pow(2.0).plusInPlace(1.0).reciprocalInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dArcTan);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        DoubleTensor inputValue = inputVertex.getValue();

        //dArcTandx = 1 / (1 + x^2)
        DoubleTensor dSelfWrtInput = inputValue.pow(2.0).plusInPlace(1.0).reciprocalInPlace();

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.multiplyAlongWrtDimensions(dSelfWrtInput));

        return partials;
    }
}
