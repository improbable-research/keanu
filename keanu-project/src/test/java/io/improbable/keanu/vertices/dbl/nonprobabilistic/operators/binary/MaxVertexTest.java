package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

public class MaxVertexTest {

    @Test
    public void maxOfTwoScalarValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 3.0, DoubleVertex::max);
    }

    @Test
    public void maxOfTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 6.0, 4.0},
            new double[]{2.0, 4.0, 3.0, 4.0},
            new double[]{2.0, 4.0, 6.0, 4.0},
            DoubleVertex::max
        );
    }

}
