package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class TensorCeilVertexTest {

    @Test
    public void ceilTwoScalarVertexValues() {
        operatesOnScalarVertexValue(
            2.8,
            3.0,
            DoubleTensorVertex::ceil
        );
    }

    @Test
    public void ceilTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{2.1, 2.8, -6.2, 4.0},
            new double[]{3.0, 3.0, -6.0, 4.0},
            DoubleTensorVertex::ceil
        );
    }

}
