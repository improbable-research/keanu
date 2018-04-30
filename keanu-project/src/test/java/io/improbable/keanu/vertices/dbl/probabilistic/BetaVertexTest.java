package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class BetaVertexTest {
    private final Logger log = LoggerFactory.getLogger(BetaVertexTest.class);

    private static final double DELTA = 0.0001;

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.01;

        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0), random);

        double mean = 0.5;
        double standardDeviation = Math.sqrt(9.0 / (36 * 7));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(N, b, mean, standardDeviation, epsilon);
    }

    @Test
    public void equalAlphaAndBetaGivesZeroGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0), random);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(0, gradient, 0);
    }

    @Test
    public void equalAlphaAndBetaGivesPositiveGradientBeforeCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0), random);
        double value = 0.25;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void equalAlphaAndBetaGivesNegativeGradientAfterCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(3.0), random);
        double value = 0.75;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaGreaterThanBetaGivesPositiveGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(3.0), new ConstantDoubleVertex(1.5), random);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaLessThanBetaGivesNegativeGradientAtCentre() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0), random);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dDensityAtValue().get(b.getId());
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(2.0), new ConstantDoubleVertex(2.0), random);
        double atValue = 0.5;
        double logOfDensity = Math.log(b.density(atValue));
        double logDensity = b.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0), random);
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(b, 0.75, 0.0001);
    }

    @Test
    public void sampleMatchesDensity() {
        BetaVertex b = new BetaVertex(new ConstantDoubleVertex(2.0), new ConstantDoubleVertex(2.0), random);

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
        BetaVertex beta = new BetaVertex(uniformA, new ConstantDoubleVertex(3.0), random);

        double vertexStartValue = 0.1;
        double vertexEndValue = 0.9;
        double vertexIncrement = 0.1;

        double alphaStart = 2.0;
        double alphaEnd = 2.5;
        double alphaIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
                alphaStart,
                alphaEnd,
                alphaIncrement,
                uniformA,
                beta,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA
        );
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0));
        BetaVertex beta = new BetaVertex(new ConstantDoubleVertex(1.0), uniformA, random);

        double vertexStartValue = 0.1;
        double vertexEndValue = 0.9;
        double vertexIncrement = 0.1;

        double betaStart = 1.0;
        double betaEnd = 1.5;
        double betaIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
                betaStart,
                betaEnd,
                betaIncrement,
                uniformA,
                beta,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA
        );
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueAlpha = 2.0;
        double trueBeta = 2.0;

        List<DoubleVertex> alphaBeta = new ArrayList<>();
        alphaBeta.add(new ConstantDoubleVertex(trueAlpha));
        alphaBeta.add(new ConstantDoubleVertex(trueBeta));

        List<DoubleVertex> latentAlphaBeta = new ArrayList<>();
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0, random));
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0, random));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new BetaVertex(hyperParams.get(0), hyperParams.get(1), random),
                alphaBeta,
                latentAlphaBeta,
                1000
        );
    }
}
