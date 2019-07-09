package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpVertex;

public class IntegerProductVertex extends IntegerUnaryOpVertex {
    private final static String OVER_DIMENSIONS = "overDimensions";
    private final int[] overDimensions;

    @ExportVertexToPythonBindings
    public IntegerProductVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex, @LoadVertexParam(OVER_DIMENSIONS) int[] overDimensions) {
        super(inputVertex.getShape(), inputVertex);
        this.overDimensions = overDimensions;
    }

    public IntegerProductVertex(IntegerVertex inputVertex) {
        super(new long[0], inputVertex);
        this.overDimensions = null;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        if (overDimensions != null) {
            return value.product();
        } else {
            return value.product(overDimensions);
        }
    }

    @SaveVertexParam(OVER_DIMENSIONS)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
