package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

public class IntegerModVertex extends IntegerBinaryOpVertex<Integer, IntegerTensor, IntegerVertex, Integer, IntegerTensor, IntegerVertex> {

    @ExportVertexToPythonBindings
    public IntegerModVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<Integer, IntegerTensor, IntegerVertex> left,
                            @LoadVertexParam(RIGHT_NAME) TensorVertex<Integer, IntegerTensor, IntegerVertex> right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.mod(r);
    }
}
