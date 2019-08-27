package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.Collections;
import java.util.Map;

/**
 * A vertex that takes on the value of the matrixDeterminant of the value of its input matrix
 * <p>
 * Gradient calculations (and thus gradient-based optimisations) will fail if the determinant is 0.
 * <p>
 * Reverse derivatives are implemented according to https://www.cs.ox.ac.uk/files/723/NA-08-01.pdf
 * <p>
 * Forward mode differentiation is not implemented due to requiring a tensor trace, which is not yet implemented
 */
public class MatrixDeterminantVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    @ExportVertexToPythonBindings
    public MatrixDeterminantVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> vertex) {
        super(Tensor.SCALAR_SHAPE, vertex, vertex.ofType());
        TensorShapeValidation.checkShapeIsSquareMatrix(vertex.getShape());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.matrixDeterminant();
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {

        final ForwardModePartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);
        final DoubleTensor dA = partial.get();
        final DoubleTensor AInverseTranspose = inputVertex.getValue().toDouble().matrixInverse();
        final DoubleTensor C = this.getValue().toDouble();
        final DoubleTensor result = C.times(AInverseTranspose.matrixMultiply(dA).diagPart().sum(-1));

        return new ForwardModePartialDerivative(partial.getWrtShape(), result);
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {

        DoubleTensor dC = derivativeOfOutputWithRespectToSelf.get();
        DoubleTensor C = this.getValue().toDouble();
        DoubleTensor dCC = dC.times(C);

        long[] dCCExpandedShape = TensorShape.concat(dCC.getShape(), new long[]{1, 1});
        DoubleTensor AInverseTranspose = inputVertex.getValue().toDouble().matrixInverse().transpose();
        ReverseModePartialDerivative toInput = new ReverseModePartialDerivative(
            derivativeOfOutputWithRespectToSelf.getOfShape(),
            dCC.reshape(dCCExpandedShape).times(AInverseTranspose)
        );

        return Collections.singletonMap(inputVertex, toInput);
    }
}
