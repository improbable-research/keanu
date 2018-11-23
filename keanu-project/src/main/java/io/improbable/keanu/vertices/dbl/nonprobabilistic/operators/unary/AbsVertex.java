package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;


import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;


public class AbsVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the absolute of a vertex
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public AbsVertex(@LoadParentVertex(INPUT_VERTEX_NAME) DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.abs();
    }
}
