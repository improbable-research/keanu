package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class AbsVertexTest {

    @Test
    public void absTwoScalarVertexValues() {
        operatesOnScalarVertexValue(
            -3,
            3.0,
            DoubleVertex::abs
        );
    }

    @Test
    public void absTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{-2, 3.0, -6.0, 4.0},
            new double[]{2.0, 3.0, 6.0, 4.0},
            DoubleVertex::abs
        );
    }

}
