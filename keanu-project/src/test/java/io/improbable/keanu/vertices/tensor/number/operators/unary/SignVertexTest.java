package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;

public class SignVertexTest {

    @Test
    public void signTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{-2, 3.0, -6.0, 4.0},
            new double[]{-1, 1, -1, 1},
            DoubleVertex::sign
        );
    }
}
