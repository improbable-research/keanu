package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ArcSinVertexTest {

    @Test
    public void asinScalarVertexValues() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.asin(Math.PI),
            DoubleVertex::asin
        );
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsAsin() {
        calculatesDerivativeOfScalar(
            0.5,
            1.0 / Math.sqrt(1.0 - 0.5 * 0.5),
            DoubleVertex::asin
        );
    }

    @Test
    public void asinMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.asin(0.0), Math.asin(0.1), Math.asin(0.2), Math.asin(0.3)},
            DoubleVertex::asin
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWiseAsin() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1.0 / Math.sqrt(1.0 - 0.1 * 0.1),
                1.0 / Math.sqrt(1.0 - 0.2 * 0.2),
                1.0 / Math.sqrt(1.0 - 0.3 * 0.3),
                1.0 / Math.sqrt(1.0 - 0.4 * 0.4)
            }),
            DoubleVertex::asin
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -0.25, 0.25);
        DoubleVertex outputVertex = inputVertex.times(2.0).asin();

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-4);
    }

}
