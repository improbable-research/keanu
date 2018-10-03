package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

public class AbsVertexTest {

    @Test
    public void absTwoScalarVertexValues() {
        operatesOnScalarVertexValue(-3, 3.0, DoubleVertex::abs);
    }

    @Test
    public void absTwoMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
                new double[] {-2, 3.0, -6.0, 4.0},
                new double[] {2.0, 3.0, 6.0, 4.0},
                DoubleVertex::abs);
    }
}
