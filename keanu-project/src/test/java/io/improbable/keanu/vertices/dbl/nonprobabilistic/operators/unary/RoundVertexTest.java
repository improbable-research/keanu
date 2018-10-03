package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

public class RoundVertexTest {

    private static final double EPSILON = 1e-10;

    @Test
    public void roundAScalarVertexValueUp() {
        operatesOnScalarVertexValue(2.5 + EPSILON, 3.0, DoubleVertex::round);
    }

    @Test
    public void roundAScalarVertexValueDown() {
        operatesOnScalarVertexValue(2.5 - EPSILON, 2.0, DoubleVertex::round);
    }

    @Test
    public void roundANegativeScalarVertexValueUp() {
        operatesOnScalarVertexValue(-2.5 + EPSILON, -2.0, DoubleVertex::round);
    }

    @Test
    public void roundANegativeScalarVertexValueDown() {
        operatesOnScalarVertexValue(-2.5 - EPSILON, -3.0, DoubleVertex::round);
    }

    @Test
    public void exactlyPointFiveGetsRoundedUp() {
        operatesOnScalarVertexValue(2.5, 3.0, DoubleVertex::round);
    }

    /**
     * NB: This is not native Java behaviour But we want it to match Python behaviour (which is what
     * ND4J uses)
     */
    @Test
    public void exactlyPointFiveGetsRoundedDownForNegatives() {
        operatesOnScalarVertexValue(-2.5, -3.0, DoubleVertex::round);
    }

    @Test
    public void roundSomeMatrixVertexValues() {
        operatesOn2x2MatrixVertexValues(
                new double[] {2.5 - EPSILON, 2.5 + EPSILON, -2.5 + EPSILON, -2.5 - EPSILON},
                new double[] {2.0, 3.0, -2.0, -3.0},
                DoubleVertex::round);
    }

    @Test
    public void withMatricesExactlyPointFiveGetsRoundedUpForPositivesButDownForNegatives() {
        operatesOn2x2MatrixVertexValues(
                new double[] {2.5, 3.5, -2.5, -3.5},
                new double[] {3.0, 4.0, -3.0, -4.0},
                DoubleVertex::round);
    }
}
