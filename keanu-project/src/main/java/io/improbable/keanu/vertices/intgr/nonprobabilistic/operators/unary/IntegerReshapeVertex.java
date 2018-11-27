package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerReshapeVertex extends IntegerUnaryOpVertex {

    private static final String SHAPE_NAME = "proposedShape";

    public IntegerReshapeVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                                @LoadVertexParam(SHAPE_NAME) long... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.reshape(getShape());
    }

    @SaveVertexParam(SHAPE_NAME)
    public long[] getShapeParam() {
        return getShape();
    }
}
