package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

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
        calculatesDualNumberOfAVectorsAndScalar(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            2,
            DoubleTensor.eye(4).times(2).reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4, 1, 1),
            DoubleVertex::multiply
        );
    }
}
