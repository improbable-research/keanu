package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.unary.UnaryOperationTestHelpers.*;

public class TensorAbsVertexTest {

    @Test
    public void absTwoScalarVertexValues() {
        operatesOnScalarVertexValue(
            -3,
            3.0,
            DoubleTensorVertex::abs
        );
    }

    @Test
    public void absTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{-2, 3.0, -6.0, 4.0},
            new double[]{2.0, 3.0, 6.0, 4.0},
            DoubleTensorVertex::abs
        );
    }

}
