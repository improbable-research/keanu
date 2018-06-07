package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

public class DifferenceVertexTest {

    @Test
    public void subtractsTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, -1, DoubleVertex::minus);
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsSubtracted() {
        calculatesDualNumberOfTwoScalars(2.0, 3.0, 1.0, -1.0, DoubleVertex::minus);
    }

    @Test
    public void subtractsTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 6.0, 4.0},
            new double[]{2.0, 4.0, 3.0, 8.0},
            new double[]{-1.0, -2.0, 3.0, -4.0},
            DoubleVertex::minus
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseSubtracted() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            new double[]{1.0, 2.0, 3.0, 4.0},
            new double[]{2.0, 3.0, 4.0, 5.0},
            new double[]{1.0, 1.0, 1.0, 1.0},
            new double[]{-1.0, -1.0, -1.0, -1.0},
            DoubleVertex::minus
        );
    }
}
