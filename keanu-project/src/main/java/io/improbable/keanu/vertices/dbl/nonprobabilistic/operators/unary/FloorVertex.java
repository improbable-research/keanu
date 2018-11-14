package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;


public class FloorVertex extends DoubleUnaryOpVertex implements SaveableVertex {

    /**
     * Applies the Floor operator to a vertex.
     * This maps a vertex to the biggest integer less than or equal to its value
     *
     * @param inputVertex the vertex to be floor'd
     */
    @ExportVertexToPythonBindings
    public FloorVertex(@LoadParentVertex(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.floor();
    }
}
