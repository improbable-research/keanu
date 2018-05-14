package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary.BinaryOperationTestHelpers.*;

public class TensorDifferenceVertexTest {

    @Test
    public void subtractsTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, -1, DoubleTensorVertex::minus);
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsSubtracted() {
        calculatesDualNumberOfTwoScalars(2.0, 3.0, 1.0, -1.0, DoubleTensorVertex::minus);
    }

    @Test
    public void subtractsTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
                new double[]{1.0, 2.0, 6.0, 4.0},
                new double[]{2.0, 4.0, 3.0, 8.0},
                new double[]{-1.0, -2.0, 3.0, -4.0},
                DoubleTensorVertex::minus
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseSubtracted() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new double[]{2.0, 3.0, 4.0, 5.0},
                new double[]{1.0, 1.0, 1.0, 1.0},
                new double[]{-1.0, -1.0, -1.0, -1.0},
                DoubleTensorVertex::minus
        );
    }
}
