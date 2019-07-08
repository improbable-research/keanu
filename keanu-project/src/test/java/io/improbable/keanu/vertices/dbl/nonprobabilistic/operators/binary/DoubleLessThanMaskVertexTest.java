package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

public class DoubleLessThanMaskVertexTest {

    @Test
    public void lessThanScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
            2.0,
            3.0,
            1.0,
            DoubleVertex::lessThanMask
        );
    }

    @Test
    public void lessThanMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 4.0, 3.0, -3.0},
            new double[]{2.0, 2.0, 3.0, 3.0},
            new double[]{1.0, 0.0, 0.0, 1.0},
            DoubleVertex::lessThanMask
        );
    }
}
