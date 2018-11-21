package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;


public class RoundVertex extends DoubleUnaryOpVertex {

    /**
     * Applies the Rounding operator to a vertex.
     * This maps a vertex to the nearest integer value
     *
     * @param inputVertex the vertex to be rounded
     */
    @ExportVertexToPythonBindings
    public RoundVertex(@LoadParentVertex(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.round();
    }
}
