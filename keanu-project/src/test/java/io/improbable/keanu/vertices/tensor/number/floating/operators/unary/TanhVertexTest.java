package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class TanhVertexTest {

    @Test
    public void tanhScalarVertexValue() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.tanh(Math.PI),
            DoubleVertex::tanh
        );
    }

    @Test
    public void calculatesDerivativeOScalarTanh() {
        calculatesDerivativeOfScalar(
            0.5,
            1 / Math.pow(Math.cos(0.5), 2),
            DoubleVertex::tanh
        );
    }

    @Test
    public void tanMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.tanh(0.0), Math.tanh(0.1), Math.tanh(0.2), Math.tanh(0.3)},
            DoubleVertex::tanh
        );
    }

    @Test
    public void calculatesDerivativeOfMatrixElementWiseTanh() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1 / Math.pow(Math.cos(0.1), 2),
                1 / Math.pow(Math.cos(0.2), 2),
                1 / Math.pow(Math.cos(0.3), 2),
                1 / Math.pow(Math.cos(0.4), 2)
            }),
            DoubleVertex::tanh
        );
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -1.0, 1.0);
        DoubleVertex outputVertex = inputVertex.div(3).tanh();

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.0001, 1e-6);
    }

}
