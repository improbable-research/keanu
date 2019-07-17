package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class SinhVertexTest {

    @Test
    public void sinhScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.sinh(Math.PI),
            DoubleVertex::sinh
        );
    }

    @Test
    public void calculatesDerivativeOScalarSinh() {
        calculatesDerivativeOfScalar(
            0.5,
            Math.cosh(0.5),
            DoubleVertex::sinh
        );
    }

    @Test
    public void sinhMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.sinh(0.0), Math.sinh(0.1), Math.sinh(0.2), Math.sinh(0.3)},
            DoubleVertex::sinh
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseSinh() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{Math.cosh(0.1), Math.cosh(0.2), Math.cosh(0.3), Math.cosh(0.4)}),
            DoubleVertex::sinh
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::sinh);
    }

}
