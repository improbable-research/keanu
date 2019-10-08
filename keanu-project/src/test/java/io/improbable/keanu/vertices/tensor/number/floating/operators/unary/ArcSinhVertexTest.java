package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ArcSinhVertexTest {

    @Test
    public void asinhScalarVertexValues() {
        operatesOnScalarVertexValue(
            Math.PI,
            FastMath.asinh(Math.PI),
            DoubleVertex::asinh
        );
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsAsinh() {
        calculatesDerivativeOfScalar(
            0.5,
            1.0 / Math.sqrt(0.5 * 0.5 + 1),
            DoubleVertex::asinh
        );
    }

    @Test
    public void asinhhMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{FastMath.asinh(0.0), FastMath.asinh(0.1), FastMath.asinh(0.2), FastMath.asinh(0.3)},
            DoubleVertex::asinh
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWiseAsinh() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1.0 / Math.sqrt(0.1 * 0.1 + 1),
                1.0 / Math.sqrt(0.2 * 0.2 + 1),
                1.0 / Math.sqrt(0.3 * 0.3 + 1),
                1.0 / Math.sqrt(0.4 * 0.4 + 1)
            }),
            DoubleVertex::asinh
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::asinh);
    }

}
