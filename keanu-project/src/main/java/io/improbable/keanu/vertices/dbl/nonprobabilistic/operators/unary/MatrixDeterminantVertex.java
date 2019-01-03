package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

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
public class MatrixDeterminantVertex extends DoubleUnaryOpVertex implements Differentiable {

    @ExportVertexToPythonBindings
    public MatrixDeterminantVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex vertex) {
        super(Tensor.SCALAR_SHAPE, vertex);
        TensorShapeValidation.checkShapeIsSquareMatrix(vertex.getShape());
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.determinant());
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        PartialDerivative dOutputTimesDeterminant = derivativeOfOutputWithRespectToSelf
            .multiplyBy(inputVertex.getValue().determinant());

        long[] resultShape = TensorShape.concat(
            derivativeOfOutputWithRespectToSelf.get().getShape(),
            inputVertex.getShape()
        );

        DoubleTensor reshapedPartial = PartialDerivative.increaseRankByAppendingOnesToShape(
            dOutputTimesDeterminant.get(),
            resultShape.length
        );

        DoubleTensor broadcastedPartial = DoubleTensor
            .zeros(resultShape)
            .plus(reshapedPartial);

        DoubleTensor inverseTranspose = inputVertex.getValue().transpose().matrixInverse();

        PartialDerivative toInput = new PartialDerivative(broadcastedPartial)
            .multiplyAlongWrtDimensions(inverseTranspose);

        return Collections.singletonMap(inputVertex, toInput);
    }
}
