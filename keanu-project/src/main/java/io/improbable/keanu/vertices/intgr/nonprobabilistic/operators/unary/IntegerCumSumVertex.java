package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerCumSumVertex extends IntegerUnaryOpVertex {
    private final static String REQUESTED_DIMENSION = "requestedDimension";
    private final int requestedDimension;

    @ExportVertexToPythonBindings
    public IntegerCumSumVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                               @LoadVertexParam(REQUESTED_DIMENSION) int requestedDimension) {
        super(inputVertex.getShape(), inputVertex);
        this.requestedDimension = requestedDimension;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.cumSum(requestedDimension);
    }

    @SaveVertexParam(REQUESTED_DIMENSION)
    public int getRequestedDimension() {
        return this.requestedDimension;
    }
}
