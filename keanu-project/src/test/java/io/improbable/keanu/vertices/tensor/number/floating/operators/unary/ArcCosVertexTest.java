package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ArcCosVertexTest {

    @Test
    public void acosScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.acos(Math.PI),
            DoubleVertex::acos
        );
    }

    @Test
    public void calculatesDerivativeOScalarACos() {
        calculatesDerivativeOfScalar(
            0.5,
            -1.0 / Math.sqrt(1.0 - 0.5 * 0.5),
            DoubleVertex::acos
        );
    }

    @Test
    public void acosMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.acos(0.0), Math.acos(0.1), Math.acos(0.2), Math.acos(0.3)},
            DoubleVertex::acos
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseACos() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                -1.0 / Math.sqrt(1.0 - 0.1 * 0.1),
                -1.0 / Math.sqrt(1.0 - 0.2 * 0.2),
                -1.0 / Math.sqrt(1.0 - 0.3 * 0.3),
                -1.0 / Math.sqrt(1.0 - 0.4 * 0.4)
            }),
            DoubleVertex::acos
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::acos);
    }

}
