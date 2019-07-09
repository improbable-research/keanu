package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.dbl.TensorMulByMatrixMul.getResultShape;

public class IntegerTensorMultiplyVertex extends IntegerBinaryOpVertex {

    private static final String DIMS_LEFT = "dimsLeft";
    private static final String DIMS_RIGHT = "dimsRight";

    private final int[] dimsLeft;
    private final int[] dimsRight;

    @ExportVertexToPythonBindings
    public IntegerTensorMultiplyVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left,
                                       @LoadVertexParam(RIGHT_NAME) IntegerVertex right,
                                       @LoadVertexParam(DIMS_LEFT) int[] dimsLeft,
                                       @LoadVertexParam(DIMS_RIGHT) int[] dimsRight) {
        super(getResultShape(left.getShape(), right.getShape(), dimsLeft, dimsRight), left, right);
        this.dimsLeft = dimsLeft;
        this.dimsRight = dimsRight;
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.tensorMultiply(r, dimsLeft, dimsRight);
    }

    @SaveVertexParam(DIMS_LEFT)
    public int[] getDimsLeft() {
        return dimsLeft;
    }

    @SaveVertexParam(DIMS_RIGHT)
    public int[] getDimsRight() {
        return dimsRight;
    }
}
