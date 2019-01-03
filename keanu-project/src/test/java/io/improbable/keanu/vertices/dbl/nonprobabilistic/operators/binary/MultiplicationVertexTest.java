package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDerivativeOfAScalarAndVector;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDerivativeOfAVectorAndScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDerivativeOfTwoMatricesElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDerivativeOfTwoScalars;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

public class MultiplicationVertexTest {

    @Test
    public void multipliesTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 6.0, DoubleVertex::multiply);
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsMultiplied() {
        calculatesDerivativeOfTwoScalars(2.0, 3.0, 3.0, 2.0, DoubleVertex::multiply);
    }

    @Test
    public void multipliesTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 3.0, 4.0},
            new double[]{2.0, 3.0, 4.0, 5.0},
            new double[]{2.0, 6.0, 12.0, 20.0},
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWiseMultiplied() {
        calculatesDerivativeOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 2, 2),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}).diag().reshape(2, 2, 2, 2),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}).diag().reshape(2, 2, 2, 2),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDerivativeOfTwoVectorsElementWiseMultiplied() {
        calculatesDerivativeOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.create(2.0, 3.0, 4.0, 5.0),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}).diag().reshape(4, 4),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}).diag().reshape(4, 4),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDerivativeOfAVectorsAndScalarMultiplied() {
        calculatesDerivativeOfAVectorAndScalar(
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            2,
            DoubleTensor.eye(4).times(2).reshape(4, 4),
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDerivativeofAScalarAndVectorsMultiplied() {
        calculatesDerivativeOfAScalarAndVector(
            2,
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.eye(4).timesInPlace(2).reshape(4, 4),
            DoubleVertex::multiply
        );
    }

    @Test
    public void finiteDifferenceMatchesScalarElementwise() {
        testWithFiniteDifference(new long[0], new long[0]);
    }

    @Test
    public void finiteDifferenceMatchesRank1Elementwise() {
        testWithFiniteDifference(new long[]{3}, new long[]{3});
    }

    @Test
    public void finiteDifferenceMatchesRank2Elementwise() {
        testWithFiniteDifference(new long[]{2, 3}, new long[]{2, 3});
    }

    @Test
    public void finiteDifferenceMatchesRank3Elementwise() {
        testWithFiniteDifference(new long[]{2, 2, 2}, new long[]{2, 2, 2});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastWithRank3AndLengthOne() {
        testWithFiniteDifference(new long[]{2, 2, 2}, new long[]{1, 1, 1});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastWithLengthOneAndRank3() {
        testWithFiniteDifference(new long[]{1, 1, 1}, new long[]{2, 2, 2});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastWithRank3AndScalar() {
        testWithFiniteDifference(new long[]{2, 2, 2}, new long[]{});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastWithScalarAndRank3() {
        testWithFiniteDifference(new long[]{}, new long[]{2, 2, 2});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastRank2() {
        testWithFiniteDifference(new long[]{2, 4}, new long[]{1, 4});
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcastRank3() {
        testWithFiniteDifference(new long[]{2, 1, 4}, new long[]{1, 1, 4});
        testWithFiniteDifference(new long[]{2, 3, 4}, new long[]{1, 3, 4});
    }

    public void testWithFiniteDifference(long[] leftShape, long[] rightShape) {
        UniformVertex A = new UniformVertex(leftShape, -10.0, 10.0);
        UniformVertex B = new UniformVertex(rightShape, -10.0, 10.0);
        MultiplicationVertex C = A.times(B);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A, B), C, 1e-6, 1e-10);
    }
}
