package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

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
    public void calculatesDualNumberOfTwoScalarsAtan() {
        calculatesDualNumberOfScalar(
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
    public void calculatesDualNumberOfTwoMatricesElementWiseAtan() {
        calculatesDualNumberOfMatrixElementWiseOperator(
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
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(3).atan();
        final double INCREMENT = 0.001;
        final double DELTA = 1e-4;

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, INCREMENT, DELTA);
    }

}
