package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShape.getPermutedResultShape;

public class IntegerPermuteVertex extends IntegerUnaryOpVertex {

    private static final String REARRANGE_NAME = "rearrange";

    private final int[] rearrange;

    @ExportVertexToPythonBindings
    public IntegerPermuteVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                                @LoadVertexParam(REARRANGE_NAME) int... rearrange) {
        super(getPermutedResultShape(inputVertex.getShape(), rearrange), inputVertex);
        this.rearrange = rearrange;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.permute(rearrange);
    }

    @SaveVertexParam(REARRANGE_NAME)
    public int[] getRearrange() {
        return rearrange;
    }
}
