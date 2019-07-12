package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;


public class AbsVertex extends DoubleUnaryOpVertex {

    /**
     * Takes the absolute of a vertex
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public AbsVertex(@LoadVertexParam(INPUT_VERTEX_NAME) Vertex<DoubleTensor, ?> inputVertex) {
        super(inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.abs();
    }
}
