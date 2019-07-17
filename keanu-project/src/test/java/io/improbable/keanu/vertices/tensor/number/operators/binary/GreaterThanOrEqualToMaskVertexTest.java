package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

public class GreaterThanOrEqualToMaskVertexTest {

    @Test
    public void greaterThanOrEqualToScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
            2.0,
            3.0,
            0.0,
            DoubleVertex::greaterThanOrEqualToMask
        );
    }

    @Test
    public void greaterThanMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 4.0, 3.0, -3.0},
            new double[]{2.0, 2.0, 3.0, 3.0},
            new double[]{0.0, 1.0, 1.0, 0.0},
            DoubleVertex::greaterThanOrEqualToMask
        );
    }
}
