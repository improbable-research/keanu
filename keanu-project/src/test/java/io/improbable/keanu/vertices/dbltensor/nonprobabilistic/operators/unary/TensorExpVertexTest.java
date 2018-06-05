package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.*;

public class TensorExpVertexTest {

    @Test
    public void expScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            Math.exp(5),
            DoubleTensorVertex::exp
        );
    }

    @Test
    public void calculatesDualNumberOScalarExp() {
        calculatesDualNumberOfScalar(
            0.5,
            Math.exp(0.5),
            DoubleTensorVertex::exp
        );
    }

    @Test
    public void expMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.exp(0.0), Math.exp(0.1), Math.exp(0.2), Math.exp(0.3)},
            DoubleTensorVertex::exp
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWiseexp() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            new double[]{Math.exp(0.1), Math.exp(0.2), Math.exp(0.3), Math.exp(0.4)},
            DoubleTensorVertex::exp
        );
    }

}
