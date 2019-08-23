package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.AutoDiffBroadcast;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.getMatrixMultiplicationResultingShape;

public class MatrixMultiplicationVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    /**
     * Matrix multiplies one vertex by another. C = AB
     *
     * @param left  vertex A
     * @param right vertex B
     */
    @ExportVertexToPythonBindings
    public MatrixMultiplicationVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                      @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(getMatrixMultiplicationResultingShape(left.getShape(), right.getShape()),
            left, right, left.ofType());
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.matrixMultiply(r);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        PartialDerivative dOutputsWrtLeft = PartialDerivative
            .matrixMultiply(
                derivativeOfOutputWithRespectToSelf,
                right.getValue().toDouble(),
                true
            );

        PartialDerivative dOutputsWrtRight = PartialDerivative
            .matrixMultiply(
                derivativeOfOutputWithRespectToSelf,
                left.getValue().toDouble(),
                false
            );

        int[] sumRight = AutoDiffBroadcast.dimensionsWithShapeChange(dOutputsWrtRight.get().getShape(), this.getRank(), right.getShape());
        int[] sumLeft = AutoDiffBroadcast.dimensionsWithShapeChange(dOutputsWrtLeft.get().getShape(), this.getRank(), left.getShape());

        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        partials.put(left, new PartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), dOutputsWrtLeft.get().sum(sumLeft)));
        partials.put(right, new PartialDerivative(derivativeOfOutputWithRespectToSelf.getOfShape(), dOutputsWrtRight.get().sum(sumRight)));

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

        return partialsFromLeft.add(partialsFromRight);
    }
}
