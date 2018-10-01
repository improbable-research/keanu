package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Collections;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

/**
 * A vertex that takes on the value of the matrixDeterminant of the value of its input matrix
 * <p>
 * Gradient calculations (and thus gradient-based optimisations) will fail if the matrixDeterminant is 0.
 */
public class MatrixDeterminantVertex extends DoubleUnaryOpVertex {
    public MatrixDeterminantVertex(DoubleVertex vertex) {
        super(Tensor.SCALAR_SHAPE, vertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return DoubleTensor.scalar(value.determinant());
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        // As this is the inverse, this will fail if the matrixDeterminant is zero
        DoubleTensor inverseTranspose = inputVertex.getValue().transpose().matrixInverse();

        // Reverse mode auto diff of matrix matrixDeterminant: https://www.cs.ox.ac.uk/files/723/NA-08-01.pdf
        // Using variables from the paper
        // Abar = derivativeOfOutputsWithRespectToInputs (what we want)
        // A = inputVertex
        // C = det(inputVertex)
        // Cbar = derivativeOfOutputsWithRespectToSelf
        // Abar_i,j = C * Cbar_i,j * inverse(transpose(A))
        PartialDerivatives derivativeOfOutputsWithRespectToInputs = derivativeOfOutputsWithRespectToSelf
            .multiplyBy(inputVertex.getValue().determinant())
            .multiplyAlongWrtDimensions(inverseTranspose, this.inputVertex.getShape());

        return Collections.singletonMap(inputVertex, derivativeOfOutputsWithRespectToInputs);
    }
}
