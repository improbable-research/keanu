package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShape.getSummationResultShape;

public class IntegerSumVertex extends IntegerUnaryOpVertex {

    private static final String DIMENSIONS_NAME = "overDimensions";

    private final int[] overDimensions;

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex    the vertex to have its values summed
     * @param overDimensions the dimensions to sum over
     */
    @ExportVertexToPythonBindings
    public IntegerSumVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                            @LoadVertexParam(DIMENSIONS_NAME) int[] overDimensions) {
        super(getSummationResultShape(inputVertex.getShape(), overDimensions), inputVertex);
        this.overDimensions = overDimensions;
    }

    public IntegerSumVertex(IntegerVertex inputVertex) {
        super(new long[0], inputVertex);
        this.overDimensions = null;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        if (overDimensions == null) {
            return IntegerTensor.scalar(value.sum());
        } else {
            return value.sum(overDimensions);
        }
    }

    @SaveVertexParam(DIMENSIONS_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
