package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

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
    public MatrixDeterminantVertex(DoubleVertex vertex) {
        super(Tensor.SCALAR_SHAPE, vertex);
        TensorShapeValidation.checkShapeIsSquareMatrix(vertex.getShape());
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.determinant());
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        DoubleTensor inverseTranspose = inputVertex.getValue().transpose().matrixInverse();

        PartialDerivatives derivativeOfOutputsWithRespectToInputs = derivativeOfOutputsWithRespectToSelf
            .multiplyBy(inputVertex.getValue().determinant())
            .multiplyAlongWrtDimensions(inverseTranspose, this.inputVertex.getShape());

        return Collections.singletonMap(inputVertex, derivativeOfOutputsWithRespectToInputs);
    }
}
