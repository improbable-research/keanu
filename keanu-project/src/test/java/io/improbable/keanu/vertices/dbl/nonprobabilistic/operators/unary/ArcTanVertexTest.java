package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

public class ArcTanVertexTest {

    @Test
    public void atanScalarVertexValues() {
        operatesOnScalarVertexValue(
            Math.PI,
            Math.atan(Math.PI),
            DoubleVertex::atan
        );
    }

    @Test
    public void calculatesDerivativeOfTwoScalarsAtan() {
        calculatesDerivativeOfScalar(
            0.5,
            1.0 / (1.0 + 0.5 * 0.5),
            DoubleVertex::atan
        );
    }

    @Test
    public void atanMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Math.atan(0.0), Math.atan(0.1), Math.atan(0.2), Math.atan(0.3)},
            DoubleVertex::atan
        );
    }

    @Test
    public void calculatesDerivativeOfTwoMatricesElementWiseAtan() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                1.0 / (1.0 + 0.1 * 0.1),
                1.0 / (1.0 + 0.2 * 0.2),
                1.0 / (1.0 + 0.3 * 0.3),
                1.0 / (1.0 + 0.4 * 0.4)
            }),
            DoubleVertex::atan
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{2, 2, 2}, -2.0, 2.0);
        ArcTanVertex outputVertex = inputVertex.times(2).atan();

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-4);
    }

}
