package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class TanVertexTest {

    @Test
    public void tanScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.tan(Math.PI),
            DoubleVertex::tan
        );
    }

    @Test
    public void calculatesDerivativeOScalarTan() {
        calculatesDerivativeOfScalar(
            0.5,
            1 / Math.pow(Math.cos(0.5), 2),
            DoubleVertex::tan
        );
    }

    @Test
    public void tanMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.tan(0.0), Math.tan(0.1), Math.tan(0.2), Math.tan(0.3)},
            DoubleVertex::tan
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseTan() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1 / Math.pow(Math.cos(0.1), 2),
                1 / Math.pow(Math.cos(0.2), 2),
                1 / Math.pow(Math.cos(0.3), 2),
                1 / Math.pow(Math.cos(0.4), 2)
            }),
            DoubleVertex::tan
        );
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -1.0, 1.0);
        DoubleVertex outputVertex = inputVertex.div(3).tan();

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.0001, 1e-6);
    }

}
