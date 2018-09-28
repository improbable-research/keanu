package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.apache.commons.math3.special.Gamma.digamma;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import org.apache.commons.math3.special.Gamma;
import org.junit.Rule;
import org.junit.Test;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class LogGammaVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void logGammaScalarVertexValue() {
        operatesOnScalarVertexValue(
            5,
            Gamma.logGamma(5),
            DoubleVertex::logGamma
        );
    }

    @Test
    public void calculatesDualNumberOScalarLogGamma() {
        calculatesDualNumberOfScalar(
            0.5,
            digamma(0.5),
            DoubleVertex::logGamma
        );
    }

    @Test
    public void logGammaMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{Gamma.logGamma(0.0), Gamma.logGamma(0.1), Gamma.logGamma(0.2), Gamma.logGamma(0.3)},
            DoubleVertex::logGamma
        );
    }

    @Test
    public void calculatesDualNumberOfMatrixElementWiseLogGamma() {
        calculatesDualNumberOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{digamma(0.1), digamma(0.2), digamma(0.3), digamma(0.4)}),
            DoubleVertex::logGamma
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex inputVertex = new UniformVertex(new int[]{2, 2, 2}, 1.0, 10.0);
        DoubleVertex outputVertex = inputVertex.div(3).logGamma();

        finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-5, true);
    }
}
