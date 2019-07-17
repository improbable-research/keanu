package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class SinVertexTest {

    @Test
    public void sinScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.sin(Math.PI),
            DoubleVertex::sin
        );
    }

    @Test
    public void calculatesDerivativeOScalarSin() {
        calculatesDerivativeOfScalar(
            0.5,
            Math.cos(0.5),
            DoubleVertex::sin
        );
    }

    @Test
    public void sinMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.sin(0.0), Math.sin(0.1), Math.sin(0.2), Math.sin(0.3)},
            DoubleVertex::sin
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWisesin() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{Math.cos(0.1), Math.cos(0.2), Math.cos(0.3), Math.cos(0.4)}),
            DoubleVertex::sin
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::sin);
    }

}
