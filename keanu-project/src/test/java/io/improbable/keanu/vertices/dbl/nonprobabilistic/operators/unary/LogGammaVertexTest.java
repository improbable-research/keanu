package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.apache.commons.math3.special.Gamma;
import org.junit.Rule;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDerivativeOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;
import static org.apache.commons.math3.special.Gamma.digamma;

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
    public void calculatesDerivativeOScalarLogGamma() {
        calculatesDerivativeOfScalar(
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
    public void calculatesDerivativeOfMatrixElementWiseLogGamma() {
        calculatesDerivativeOfMatrixElementWiseOperator(
            new double[]{0.1, 0.2, 0.3, 0.4},
            toDiagonalArray(new double[]{digamma(0.1), digamma(0.2), digamma(0.3), digamma(0.4)}),
            DoubleVertex::logGamma
        );
    }

    @Test
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::logGamma);
    }
}
