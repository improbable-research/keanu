package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerSumVertex extends IntegerUnaryOpVertex {

    private static final String DIMENSION_NAME = "overDimensions";
    private final int[] overDimensions;

    /**
    * Performs a sum across specified dimensions. Negative dimension indexing is not supported.
    *
    * @param inputVertex    the vertex to have its values summed
    * @param overDimensions dimensions to sum over
    */
    public IntegerSumVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                            @LoadVertexParam(DIMENSION_NAME) int[] overDimensions) {
        super(inputVertex.getShape(), inputVertex);
        this.overDimensions = overDimensions;
    }

    /**
    * Performs a sum across all dimensions
    *
    * @param inputVertex the vertex to have its values summed
    */
    @ExportVertexToPythonBindings
    public IntegerSumVertex(IntegerVertex inputVertex) {
        this(inputVertex, TensorShape.dimensionRange(0, inputVertex.getRank()));
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.sum(overDimensions);
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
