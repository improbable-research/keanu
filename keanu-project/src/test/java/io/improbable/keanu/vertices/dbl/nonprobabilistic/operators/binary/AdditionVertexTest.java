package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class AdditionVertexTest {

    @Test
    public void addsTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
            2.0,
            3.0,
            5.0,
            DoubleVertex::plus
        );
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsAdded() {
        calculatesDerivativeOfTwoScalars(
            2.0,
            3.0,
            1.0,
            1.0,
            DoubleVertex::plus
        );
    }

    @Test
    public void addsTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 6.0, 4.0},
            new double[]{2.0, 4.0, 3.0, 8.0},
            new double[]{3.0, 6.0, 9.0, 12.0},
            DoubleVertex::plus
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWiseAdded() {
        calculatesDerivativeOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 2, 2),
            DoubleTensor.eye(4).reshape(2, 2, 2, 2),
            DoubleTensor.eye(4).reshape(2, 2, 2, 2),
            DoubleVertex::plus
        );
    }

    @Test
    public void calculatesDerivativeOfAVectorsAndScalarAdded() {
        calculatesDerivativeOfAVectorAndScalar(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            2,
            DoubleTensor.eye(4).reshape(1, 4, 1, 4),
            DoubleTensor.ones(1, 4, 1, 1),
            DoubleVertex::plus
        );
    }

    @Test
    public void calculatesDerivativeofAScalarAndVectorsAdded() {
        calculatesDerivativeOfAScalarAndVector(
            2,
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            DoubleTensor.ones(1, 4, 1, 1),
            DoubleTensor.eye(4).reshape(1, 4, 1, 4),
            DoubleVertex::plus
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex A = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex B = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex C = A.plus(B);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A, B), C, 1e-6, 1e-10);
    }
}
