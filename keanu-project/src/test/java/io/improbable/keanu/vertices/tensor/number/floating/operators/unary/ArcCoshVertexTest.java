package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ArcCoshVertexTest {

    @Test
    public void acoshScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            FastMath.acosh(Math.PI),
            DoubleVertex::acosh
        );
    }

    @Test
    public void calculatesDerivativeOScalarACosh() {
        calculatesDerivativeOfScalar(
            0.5,
            -1.0 / Math.sqrt(1.0 - 0.5 * 0.5),
            DoubleVertex::acosh
        );
    }

    @Test
    public void acoshMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{FastMath.acosh(0.0), FastMath.acosh(0.1), FastMath.acosh(0.2), FastMath.acosh(0.3)},
            DoubleVertex::acosh
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseACosh() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                -1.0 / Math.sqrt(1.0 - 0.1 * 0.1),
                -1.0 / Math.sqrt(1.0 - 0.2 * 0.2),
                -1.0 / Math.sqrt(1.0 - 0.3 * 0.3),
                -1.0 / Math.sqrt(1.0 - 0.4 * 0.4)
            }),
            DoubleVertex::acosh
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::acosh);
    }

}
