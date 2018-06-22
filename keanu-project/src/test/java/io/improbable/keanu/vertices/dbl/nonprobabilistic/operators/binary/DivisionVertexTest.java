package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

public class DivisionVertexTest {

    @Test
    public void dividesTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
            12.0,
            3.0,
            4.0,
            DoubleVertex::divideBy
        );
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsDivided() {
        calculatesDualNumberOfTwoScalars(
            2.0,
            3.0,
            1.0 / 3.0,
            -2.0 / 9.0,
            DoubleVertex::divideBy
        );
    }

    @Test
    public void dividesTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 6.0, 4.0},
            new double[]{2.0, 4.0, 3.0, 8.0},
            new double[]{1.0 / 2.0, 2.0 / 4.0, 6.0 / 3.0, 4.0 / 8.0},
            DoubleVertex::divideBy
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseDivided() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 1, 4),
            DoubleTensor.create(new double[]{1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{-1.0 / 4.0, -2.0 / 9.0, -3.0 / 16.0, -4.0 / 25.0}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::divideBy
        );
    }
}
