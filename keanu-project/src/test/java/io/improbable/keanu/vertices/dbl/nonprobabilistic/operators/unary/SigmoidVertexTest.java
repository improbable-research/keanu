package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class SigmoidVertexTest {

    private Sigmoid sigmoid;

    @Before
    public void setup() {
        sigmoid = new Sigmoid();
    }

    @Test
    public void sigmoidScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            sigmoid.value(5),
            DoubleVertex::sigmoid
        );
    }

    @Test
    public void calculatesDualNumberOScalarSigmoid() {
        calculatesDualNumberOfScalar(
            0.5,
            Math.exp(0.5) / Math.pow(Math.exp(0.5) + 1., 2),
            DoubleVertex::sigmoid
        );
    }

    @Test
    public void sigmoidMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{sigmoid.value(0.0), sigmoid.value(0.1), sigmoid.value(0.2), sigmoid.value(0.3)},
            DoubleVertex::sigmoid
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWisesigmoid() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{
                Math.exp(0.1) / Math.pow(Math.exp(0.1) + 1., 2),
                Math.exp(0.2) / Math.pow(Math.exp(0.2) + 1., 2),
                Math.exp(0.3) / Math.pow(Math.exp(0.3) + 1., 2),
                Math.exp(0.4) / Math.pow(Math.exp(0.4) + 1., 2)
            }),
            DoubleVertex::sigmoid
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.times(3).sigmoid();

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

}
