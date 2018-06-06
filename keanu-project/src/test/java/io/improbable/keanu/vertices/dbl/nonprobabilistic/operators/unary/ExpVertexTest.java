package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.*;

public class ExpVertexTest {

    @Test
    public void expScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            Math.exp(5),
            DoubleVertex::exp
        );
    }

    @Test
    public void calculatesDualNumberOScalarExp() {
        calculatesDualNumberOfScalar(
            0.5,
            Math.exp(0.5),
            DoubleVertex::exp
        );
    }

    @Test
    public void expMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.exp(0.0), Math.exp(0.1), Math.exp(0.2), Math.exp(0.3)},
            DoubleVertex::exp
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWiseexp() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            new double[]{Math.exp(0.1), Math.exp(0.2), Math.exp(0.3), Math.exp(0.4)},
            DoubleVertex::exp
        );
    }

}
