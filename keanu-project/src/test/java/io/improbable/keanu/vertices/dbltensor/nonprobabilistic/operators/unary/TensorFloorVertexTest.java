package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class TensorFloorVertexTest {

    @Test
    public void floorTwoScalarVertexValues() {
        operatesOnScalarVertexValue(
            2.8,
            2.0,
            DoubleTensorVertex::floor
        );
    }

    @Test
    public void floorTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{2.1, 2.8, -6.2, 4.0},
            new double[]{2.0, 2.0, -7.0, 4.0},
            DoubleTensorVertex::floor
        );
    }

}
