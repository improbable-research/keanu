package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ExpVertexTest {

    @Test
    public void expScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            Math.exp(5),
            DoubleVertex::exp
        );
    }

    @Test
    public void calculatesDerivativeOScalarExp() {
        calculatesDerivativeOfScalar(
            0.5,
            Math.exp(0.5),
            DoubleVertex::exp
        );
    }

    @Test
    public void expMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.exp(0.0), Math.exp(0.1), Math.exp(0.2), Math.exp(0.3)},
            DoubleVertex::exp
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseexp() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{Math.exp(0.1), Math.exp(0.2), Math.exp(0.3), Math.exp(0.4)}),
            DoubleVertex::exp
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::exp);
    }

}
