package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

public class RoundVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the Rounding operator to a vertex.
     * This maps a vertex to the nearest integer value
     *
     * @param inputVertex the vertex to be rounded
     */
    @ExportVertexToPythonBindings
    public RoundVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.round();
    }
}
