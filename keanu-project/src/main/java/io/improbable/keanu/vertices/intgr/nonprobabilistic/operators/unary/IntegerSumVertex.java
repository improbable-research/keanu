package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerSumVertex extends IntegerUnaryOpVertex {

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex the vertex to have its values summed
     */
    @ExportVertexToPythonBindings
    public IntegerSumVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return IntegerTensor.scalar(value.sum());
    }
}
