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
    public void calculatesDualNumberOScalarACos() {
        calculatesDualNumberOfScalar(
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
    public void calculatesDualNumberOfMatrixElementWiseACos() {
        calculatesDualNumberOfMatrixElementWiseOperator(
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
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(3).acos();
        final double INCREMENT = 0.001;
        final double DELTA = 1e-4;

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, INCREMENT, DELTA);
    }

}
