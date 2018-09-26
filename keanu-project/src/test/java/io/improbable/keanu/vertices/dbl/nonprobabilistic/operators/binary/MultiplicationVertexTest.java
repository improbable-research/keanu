package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAScalarAndVector;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAVectorAndScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoMatricesElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoScalars;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class MultiplicationVertexTest {

    @Test
    public void multipliesTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 6.0, DoubleVertex::multiply);
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsMultiplied() {
        calculatesDualNumberOfTwoScalars(2.0, 3.0, 3.0, 2.0, DoubleVertex::multiply);
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
    public void calculatesDualNumberOfTwoMatricesElementWiseMultiplied() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 2, 2),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 2, 2),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}).diag().reshape(2, 2, 2, 2),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}).diag().reshape(2, 2, 2, 2),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDualNumberOfTwoVectorsElementWiseMultiplied() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 1, 4),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDualNumberOfAVectorsAndScalarMultiplied() {
        calculatesDualNumberOfAVectorAndScalar(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            2,
            DoubleTensor.eye(4).times(2).reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4, 1, 1),
            DoubleVertex::multiply
        );
    }

    @Test
    public void calculatesDualNumberofAScalarAndVectorsMultiplied() {
        calculatesDualNumberOfAScalarAndVector(
            2,
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}).reshape(1, 4, 1, 1),
            DoubleTensor.eye(4).timesInPlace(2).reshape(1, 4, 1, 4),
            DoubleVertex::multiply
        );
    }
}
