package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static org.junit.Assert.assertArrayEquals;

public class MatrixMultiplicationVertexTest {

    @Test
    public void canSimpleMatrixMultiply() {
        DoubleTensor matrixA = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        DoubleTensor matrixB = DoubleTensor.create(new double[]{2, 4, 6, 8}, 2, 2);

        MatrixMultiplicationVertex mmul = new MatrixMultiplicationVertex(ConstantVertex.of(matrixA), ConstantVertex.of(matrixB));

        DoubleTensor mmulResult = mmul.lazyEval();

        assertArrayEquals(new double[]{14, 20, 30, 44}, mmulResult.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canAutoDiffMatrixMultiply() {

        DoubleVertex x = new UniformVertex(0, 10);
        x.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex y = new UniformVertex(0, 10);
        y.setValue(DoubleTensor.create(new double[]{2, 4, 6, 8}, 2, 2));

        DoubleVertex z = x.matrixMultiply(y);

        DoubleTensor dzdx = z.getDualNumber().getPartialDerivatives().withRespectTo(x);
        DoubleTensor dzdy = z.getDualNumber().getPartialDerivatives().withRespectTo(y);

        assertArrayEquals(
            dzdx.asFlatDoubleArray(),
            toDiagonalArray(new double[]{2, 4, 6, 8}),
            0.0
        );

        assertArrayEquals(
            dzdy.asFlatDoubleArray(),
            toDiagonalArray(new double[]{1, 2, 3, 4}),
            0.0
        );

    }
}
