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

public class LogGammaVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Returns the log of the gamma of the inputVertex
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public LogGammaVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.logGamma();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInputs) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInputs.get(inputVertex);
        return derivativeOfParentWithRespectToInputs.multiplyAlongOfDimensions(inputVertex.getValue().digamma(), getShape());
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputsWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        PartialDerivative dOutputsWrtInputVertex =
            derivativeOfOutputsWithRespectToSelf.multiplyAlongWrtDimensions(inputVertex.getValue().digamma(), getShape());
        partials.put(inputVertex, dOutputsWrtInputVertex);
        return partials;
    }
}
