package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class BetaVertexTest {
    private final Logger log = LoggerFactory.getLogger(BetaVertexTest.class);

    private static final double DELTA = 0.0001;

    @Test
    public void equalAlphaAndBetaGivesZeroGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0));
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(0, gradient, 0);
    }

    @Test
    public void equalAlphaAndBetaGivesPositiveGradientBeforeCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0));
        double value = 0.25;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void equalAlphaAndBetaGivesNegativeGradientAfterCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0));
        double value = 0.75;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaGreaterThanBetaGivesPositiveGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(1.5));
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaLessThanBetaGivesNegativeGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0));
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(2.0), new ConstantDoubleVertex(2.0));
        double atValue = 0.5;
        double logOfDensity = Math.log(b.density(atValue));
        double logDensity = b.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0));
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(b, 0.75, 0.0001);
    }

    @Test
    public void sampleMatchesDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(2.0), new ConstantDoubleVertex(2.0), new Random(1));

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(
                b,
                1000000,
                0.1,
                0.9,
                .05,
                0.01
        );
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0));
        BetaVertex beta = new BetaVertex(uniformA, new ConstantDoubleVertex(3.0));

        double vertexStartValue = 0.1;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(2.0,
                2.5,
                0.1,
                uniformA,
                beta,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0));
        BetaVertex beta = new BetaVertex(new ConstantDoubleVertex(1.0), uniformA);

        double vertexStartValue = 0.1;
        double vertexEndValue = 0.9;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                1.5,
                0.1,
                uniformA,
                beta,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }
}
