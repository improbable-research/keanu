package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerAbsVertex extends IntegerUnaryOpVertex {

    /**
     * Takes the absolute value of a vertex
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public IntegerAbsVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.abs();
    }
}
