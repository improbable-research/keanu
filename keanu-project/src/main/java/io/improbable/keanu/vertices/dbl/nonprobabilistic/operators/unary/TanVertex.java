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

public class TanVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Takes the tangent of a vertex. Tan(vertex).
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public TanVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.tan();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);
        DoubleTensor dTan = inputVertex.getValue().cos().powInPlace(2).reciprocalInPlace();
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(dTan, this.getValue().getShape());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        //dTandInput = sec^2(x)
        DoubleTensor dTandInput = inputVertex.getValue().cos().powInPlace(2).reciprocalInPlace();

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(dTandInput, this.getShape()));
        return partials;
    }
}
