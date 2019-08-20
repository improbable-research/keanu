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
        partials.put(left, new PartialDerivative(dOutputsWrtLeft.get().sum(sumLeft)));
        partials.put(right, new PartialDerivative(dOutputsWrtRight.get().sum(sumRight)));

        return partials;
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dLeftWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(left, PartialDerivative.EMPTY);
        PartialDerivative dRightWrtInput = derivativeOfParentsWithRespectToInput.getOrDefault(right, PartialDerivative.EMPTY);

        // dc = A * db + da * B;
        PartialDerivative partialsFromLeft = PartialDerivative.matrixMultiplyAlongOfDimensions(
            dLeftWrtInput,
            right.getValue().toDouble(),
            true,
            left.getShape(),
            this.getShape().length
        );

        PartialDerivative partialsFromRight = PartialDerivative.matrixMultiplyAlongOfDimensions(
            dRightWrtInput,
            left.getValue().toDouble(),
            false,
            right.getShape(),
            this.getShape().length
        );

        return partialsFromLeft.add(partialsFromRight);
    }
}
