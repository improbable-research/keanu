package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.tensor.TensorVertex;

public class NaNArgMinVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends IntegerUnaryOpVertex<T, TENSOR, VERTEX> {

    private static final String AXIS = "axis";
    private final Integer axis;

    public NaNArgMinVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        this(inputVertex, null);
    }

    @ExportVertexToPythonBindings
    public NaNArgMinVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                           @LoadVertexParam(AXIS) Integer axis) {
        super(inputVertex);
        this.axis = axis;
    }

    @Override
    protected IntegerTensor op(TENSOR value) {
        if (axis != null) {
            return value.nanArgMin(axis);
        } else {
            return value.nanArgMin();
        }
    }

    @SaveVertexParam(AXIS)
    public Integer getAxis() {
        return axis;
    }
}
