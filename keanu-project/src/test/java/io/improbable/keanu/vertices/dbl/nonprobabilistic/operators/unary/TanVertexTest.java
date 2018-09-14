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
    public void calculatesDualNumberOScalarTan() {
        calculatesDualNumberOfScalar(
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
    public void calculatesDualNumberOfMatrixElementWiseTan() {
        calculatesDualNumberOfMatrixElementWiseOperator(
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
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, -1.0, 1.0);
        DoubleVertex outputVertex = inputVertex.div(3).tan();
        final double INCREMENT = 0.0001;
        final double DELTA = 1e-5;

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, INCREMENT, DELTA);
    }

}
