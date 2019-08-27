package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.dbl.TensorMulByMatrixMul;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;

public class TensorMultiplicationVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String DIMS_LEFT = "dimsLeft";
    private static final String DIMS_RIGHT = "dimsRight";

    private final int[] dimsLeft;
    private final int[] dimsRight;

    /**
     * Tensor multiplies one vertex by another. C = AB.
     *
     * @param left      the left vertex for operand
     * @param right     the right vertex for operand
     * @param dimsLeft  The dimensions of the left for multiplying. The left shape at these dimensions must align with the
     *                  shape of the corresponding right vertex at its specified dimensions.
     * @param dimsRight The dimensions of the right for multiplying. The right shape at these dimensions must align with the
     *                  shape of the corresponding left vertex at its specified dimensions.
     */
    @ExportVertexToPythonBindings
    public TensorMultiplicationVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                      @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right,
                                      @LoadVertexParam(DIMS_LEFT) int[] dimsLeft,
                                      @LoadVertexParam(DIMS_RIGHT) int[] dimsRight) {
        super(TensorMulByMatrixMul.getResultShape(left.getShape(), right.getShape(), dimsLeft, dimsRight),
            left, right, left.ofType());
        this.dimsLeft = dimsLeft;
        this.dimsRight = dimsRight;
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.tensorMultiply(r, dimsLeft, dimsRight);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        throw new UnsupportedOperationException();
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        throw new UnsupportedOperationException();
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
