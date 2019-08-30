package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.getMatrixMultiplicationResultingShape;

public class MatrixMultiplicationVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String TRANSPOSE_LEFT = "transposeLeft";
    private static final String TRANSPOSE_RIGHT = "transposeRight";

    private final boolean transposeLeft;
    private final boolean transposeRight;

    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left           vertex A
     * @param right          vertex B
     * @param transposeLeft  transpose the left operand before multiply
     * @param transposeRight transpose the right operand before multiply
     */
    @ExportVertexToPythonBindings
    public MatrixMultiplicationVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                      @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right,
                                      @LoadVertexParam(TRANSPOSE_LEFT) boolean transposeLeft,
                                      @LoadVertexParam(TRANSPOSE_RIGHT) boolean transposeRight) {
        super(getMatrixMultiplicationResultingShape(left.getShape(), right.getShape(), transposeLeft, transposeRight),
            left, right, left.ofType());
        this.transposeLeft = transposeLeft;
        this.transposeRight = transposeRight;
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.matrixMultiply(r, transposeLeft, transposeRight);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {

        ReverseModePartialDerivative dOutputsWrtLeft = ReverseModePartialDerivative
            .matrixMultiply(
                derivativeOfOutputWithRespectToSelf,
                right.getValue().toDouble(),
                true,
                false,
                true
            );

        ReverseModePartialDerivative dOutputsWrtRight = ReverseModePartialDerivative
            .matrixMultiply(
                derivativeOfOutputWithRespectToSelf,
                left.getValue().toDouble(),
                false,
                false,
                true
            );

        int[] sumRight = AutoDiffBroadcast.dimensionsWithShapeChange(dOutputsWrtRight.get().getShape(), this.getRank(), right.getShape());
        int[] sumLeft = AutoDiffBroadcast.dimensionsWithShapeChange(dOutputsWrtLeft.get().getShape(), this.getRank(), left.getShape());

        Map<Vertex, ReverseModePartialDerivative> partials = new HashMap<>();
        partials.put(left, new ReverseModePartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), dOutputsWrtLeft.get().sum(sumLeft)));
        partials.put(right, new ReverseModePartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), dOutputsWrtRight.get().sum(sumRight)));

        return partials;
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        ForwardModePartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, ForwardModePartialDerivative.EMPTY);
        ForwardModePartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, ForwardModePartialDerivative.EMPTY);

        // dc = A * db + da * B;
        ForwardModePartialDerivative partialsFromLeft = ForwardModePartialDerivative.matrixMultiply(
            dLeftWrtInput,
            right.getValue().toDouble(),
            true
        );

        ForwardModePartialDerivative partialsFromRight = ForwardModePartialDerivative.matrixMultiply(
            dRightWrtInput,
            left.getValue().toDouble(),
            false
        );

        return partialsFromLeft.add(partialsFromRight, this.getShape());
    }

    @SaveVertexParam(TRANSPOSE_LEFT)
    public boolean isTransposeLeft() {
        return transposeLeft;
    }

    @SaveVertexParam(TRANSPOSE_RIGHT)
    public boolean isTransposeRight() {
        return transposeRight;
    }
}
