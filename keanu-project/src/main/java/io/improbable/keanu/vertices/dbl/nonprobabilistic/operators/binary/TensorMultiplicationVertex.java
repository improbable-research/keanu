package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TensorMultiplicationVertex extends DoubleBinaryOpVertex {

    private final static String LEFT_DIMS = "leftDims";
    private final static String RIGHT_DIMS = "rightDims";

    private final int[] dimsLeft;
    private final int[] dimsRight;

    @ExportVertexToPythonBindings
    public TensorMultiplicationVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                                      @LoadVertexParam(RIGHT_NAME) DoubleVertex right,
                                      @LoadVertexParam(LEFT_DIMS) int[] dimsLeft,
                                      @LoadVertexParam(RIGHT_DIMS) int[] dimsRight) {
        super(TensorShapeValidation.getTensorMultiplyResultShape(left.getShape(), right.getShape(), dimsLeft, dimsRight),
            left, right);
        this.dimsLeft = dimsLeft;
        this.dimsRight = dimsRight;
    }

    @Override
    protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
        return l.tensorMultiply(r, dimsLeft, dimsRight);
    }

    @SaveVertexParam(LEFT_DIMS)
    public int[] getDimsLeft() {
        return dimsLeft;
    }

    @SaveVertexParam(RIGHT_DIMS)
    public int[] getDimsRight() {
        return dimsRight;
    }
}
