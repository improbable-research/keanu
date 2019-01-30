package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

public class PowerVertexTest {

    @Test
    public void powerTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 8.0, DoubleVertex::pow);
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsPower() {
        calculatesDerivativeOfTwoScalars(2.0, 3.0, 3. * 4., Math.log(2.) * 8., DoubleVertex::pow);
    }

    @Test
    public void powerTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 3.0, 4.0},
            new double[]{2.0, 3.0, 4.0, 5.0},
            new double[]{1.0, 8.0, 81.0, 1024.0},
            DoubleVertex::pow
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWisePower() {
        calculatesDerivativeOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.create(2.0, 3.0, 4.0, 5.0),
            DoubleTensor.create(2.0, 3.0 * 4, 4.0 * 27, 5.0 * 256).diag().reshape(4, 4),
            DoubleTensor.create(Math.log(1.0) * 1, Math.log(2.0) * 8, Math.log(3.0) * 81, Math.log(4.0) * 1024).diag().reshape(4, 4),
            DoubleVertex::pow
        );
    }

    @Test
    public void calculatesDerivativeOfAVectorsAndScalarPower() {
        calculatesDerivativeOfAVectorAndScalar(
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            3,
            DoubleTensor.create(3.0, 3.0 * 4., 3.0 * 9, 3.0 * 16).diag().reshape(4, 4),
            DoubleTensor.create(Math.log(1.0), Math.log(2.0) * 8, Math.log(3.0) * 27, Math.log(4.0) * 64),
            DoubleVertex::pow
        );
    }

    @Test
    public void calculatesDerivativeofAScalarAndVectorPower() {
        calculatesDerivativeOfAScalarAndVector(
            3,
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.create(1., 2.0 * 3, 3.0 * 9., 4.0 * 27),
            DoubleTensor.create(Math.log(3.0) * 3, Math.log(3.0) * 9, Math.log(3.0) * 27, Math.log(3.0) * 81).diag().reshape(4, 4),
            DoubleVertex::pow
        );
    }

    @Test
    public void canCalculateWrtBaseWhenBaseIsZero() {
        UniformVertex A = new UniformVertex(-10.0, 10.0);
        A.setValue(0.0);
        UniformVertex B = new UniformVertex(-10.0, 10.0);
        B.setValue(2.0);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A,B), A.pow(B), 1e-10, 1e-10);
    }

    @Test
    public void matchesFiniteDifferenceWhenWrtExponentExists() {
        testWithFiniteDifference(DoubleTensor.scalar(5.0), DoubleTensor.scalar(0.0));
        testWithFiniteDifference(DoubleTensor.create(4.0, 1.0, 2.0, 3.0), DoubleTensor.scalar(0.0));
        testWithFiniteDifference(DoubleTensor.create(4.0, 1.0, 2.0, 3.0), DoubleTensor.scalar(2.0));
        testWithFiniteDifference(DoubleTensor.create(new double[]{4, 1, 2, 3}, 2, 2), DoubleTensor.scalar(2.0));
    }

    @Test
    public void matchesFiniteDifferenceWithPowerMatchesBaseShape() {
        testWithFiniteDifference(DoubleTensor.create(4, 1, 2, 3), DoubleTensor.create(2, 1, 2, 3));
    }

    private void testWithFiniteDifference(DoubleTensor baseValue,
                                          DoubleTensor exponentValue) {

        UniformVertex A = new UniformVertex(baseValue.getShape(), -10.0, 10.0);
        A.setValue(baseValue);
        UniformVertex B = new UniformVertex(exponentValue.getShape(), -10.0, 10.0);
        B.setValue(exponentValue);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(A, B), A.pow(B), 1e-10, 1e-10);
    }

}
