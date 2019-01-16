package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

public class PermuteVertex extends DoubleUnaryOpVertex implements Differentiable {

    private final static String REARRANGE_NAME = "rearrange";

    private final int[] rearrange;

    @ExportVertexToPythonBindings
    public PermuteVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                         @LoadVertexParam(REARRANGE_NAME) int... rearrange) {
        super(inputVertex);
        this.rearrange = rearrange;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.permute(rearrange);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        return derivativeOfParentWithRespectToInputs.permute(rearrange);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.permute(rearrange));
        return partials;
    }
}
