package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.*;

public class ArcTanVertexTest {

    @Test
    public void atanScalarVertexValues() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.atan(Math.PI),
            DoubleVertex::atan
        );
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsAtan() {
        calculatesDualNumberOfScalar(
            0.5,
            1.0 / (1.0 + 0.5 * 0.5),
            DoubleVertex::atan
        );
    }

    @Test
    public void atanMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.atan(0.0), Math.atan(0.1), Math.atan(0.2), Math.atan(0.3)},
            DoubleVertex::atan
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseAtan() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1.0 / (1.0 + 0.1 * 0.1),
                1.0 / (1.0 + 0.2 * 0.2),
                1.0 / (1.0 + 0.3 * 0.3),
                1.0 / (1.0 + 0.4 * 0.4)
            }),
            DoubleVertex::atan
        );
    }
}
