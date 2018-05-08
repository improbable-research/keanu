package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary.BinaryOperationHelper.*;

public class TensorDivisionVertexTest {

    @Test
    public void dividesTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
                12.0,
                3.0,
                4.0,
                DoubleTensorVertex::divideBy
        );
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsDivided() {
        calculatesDualNumberOfTwoScalars(
                2.0,
                3.0,
                1.0 / 3.0,
                -2.0 / 9.0,
                DoubleTensorVertex::divideBy
        );
    }

    @Test
    public void dividesTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
                new double[]{1.0, 2.0, 6.0, 4.0},
                new double[]{2.0, 4.0, 3.0, 8.0},
                new double[]{1.0 / 2.0, 2.0 / 4.0, 6.0 / 3.0, 4.0 / 8.0},
                DoubleTensorVertex::divideBy
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseDivided() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new double[]{2.0, 3.0, 4.0, 5.0},
                new double[]{1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0},
                new double[]{-1.0 / 4.0, -2.0 / 9.0, -3.0 / 16.0, -4.0 / 25.0},
                DoubleTensorVertex::divideBy
        );
    }
}
