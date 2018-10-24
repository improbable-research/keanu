package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

public class FloorVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the Floor operator to a vertex.
     * This maps a vertex to the biggest integer less than or equal to its value
     *
     * @param inputVertex the vertex to be floor'd
     */
    @ExportVertexToPythonBindings
    public FloorVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.floor();
    }

    @Override
    protected  PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        throw new UnsupportedOperationException();
    }
}
