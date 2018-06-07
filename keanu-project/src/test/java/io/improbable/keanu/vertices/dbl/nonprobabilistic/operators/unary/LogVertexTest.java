package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.*;

public class LogVertexTest {

    @Test
    public void logScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            Math.log(5),
            DoubleVertex::log
        );
    }

    @Test
    public void calculatesDualNumberOScalarLog() {
        calculatesDualNumberOfScalar(
            0.5,
            1./0.5,
            DoubleVertex::log
        );
    }

    @Test
    public void logMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.log(0.0), Math.log(0.1), Math.log(0.2), Math.log(0.3)},
            DoubleVertex::log
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWiselog() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            new double[]{1/0.1, 1/0.2, 1/0.3, 1/0.4},
            DoubleVertex::log
        );
    }

}
