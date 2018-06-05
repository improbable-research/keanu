package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.*;

public class TensorSinVertexTest {

    @Test
    public void sinScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.sin(Math.PI),
            DoubleTensorVertex::sin
        );
    }

    @Test
    public void calculatesDualNumberOScalarSin() {
        calculatesDualNumberOfScalar(
            0.5,
            Math.cos(0.5),
            DoubleTensorVertex::sin
        );
    }

    @Test
    public void sinMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.sin(0.0), Math.sin(0.1), Math.sin(0.2), Math.sin(0.3)},
            DoubleTensorVertex::sin
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWisesin() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            new double[]{Math.cos(0.1), Math.cos(0.2), Math.cos(0.3), Math.cos(0.4)},
            DoubleTensorVertex::sin
        );
    }
    
}
