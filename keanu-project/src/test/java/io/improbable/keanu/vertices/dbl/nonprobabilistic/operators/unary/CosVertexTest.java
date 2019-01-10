package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class CosVertexTest {

    @Test
    public void cosScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.cos(Math.PI),
            DoubleVertex::cos
        );
    }

    @Test
    public void calculatesDerivativeOScalarCos() {
        calculatesDerivativeOfScalar(
            0.5,
            -Math.sin(0.5),
            DoubleVertex::cos
        );
    }

    @Test
    public void cosMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.cos(0.0), Math.cos(0.1), Math.cos(0.2), Math.cos(0.3)},
            DoubleVertex::cos
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseCos() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{-Math.sin(0.1), -Math.sin(0.2), -Math.sin(0.3), -Math.sin(0.4)}),
            DoubleVertex::cos
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::cos);
    }

}
