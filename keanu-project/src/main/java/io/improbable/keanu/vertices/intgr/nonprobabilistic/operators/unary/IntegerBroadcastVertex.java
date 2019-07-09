package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerBroadcastVertex extends IntegerUnaryOpVertex {

    private static final String TO_SHAPE_NAME = "toShape";
    private final long[] toShape;

    @ExportVertexToPythonBindings
    public IntegerBroadcastVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                                  @LoadVertexParam(TO_SHAPE_NAME) long[] toShape) {
        super(toShape, inputVertex);
        this.toShape = toShape;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.broadcast(toShape);
    }

    @SaveVertexParam(TO_SHAPE_NAME)
    public long[] getToShape() {
        return toShape;
    }
}
