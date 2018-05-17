package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class BetaVertexTest {
    private final Logger log = LoggerFactory.getLogger(BetaVertexTest.class);

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.01;

        BetaVertex betaVertex = new BetaVertex(3.0, 3.0);

        double mean = 0.5;
        double standardDeviation = Math.sqrt(9.0 / (36 * 7));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
                N,
                betaVertex,
                mean,
                standardDeviation,
                epsilon,
                random
        );
    }

    @Test
    public void equalAlphaAndBetaGivesZeroGradientAtCentre() {
        BetaVertex b = new BetaVertex(3.0, 3.0);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dLogProbAtValue().get(b.getId()).scalar();
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(0, gradient, 0);
    }

    @Test
    public void equalAlphaAndBetaGivesPositiveGradientBeforeCentre() {
        BetaVertex b = new BetaVertex(3.0, 3.0);
        double value = 0.25;
        b.setValue(value);
        double gradient = b.dLogProbAtValue().get(b.getId()).scalar();
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void equalAlphaAndBetaGivesNegativeGradientAfterCentre() {
        BetaVertex b = new BetaVertex(3.0, 3.0);
        double value = 0.75;
        b.setValue(value);
        double gradient = b.dLogProbAtValue().get(b.getId()).scalar();
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaGreaterThanBetaGivesPositiveGradientAtCentre() {
        BetaVertex b = new BetaVertex(3.0, 1.5);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dLogProbAtValue().get(b.getId()).scalar();
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(1, Math.signum(gradient), 0);
    }

    @Test
    public void alphaLessThanBetaGivesNegativeGradientAtCentre() {
        BetaVertex b = new BetaVertex(1.5, 3.0);
        double value = 0.5;
        b.setValue(value);
        double gradient = b.dLogProbAtValue().get(b.getId()).scalar();
        log.info("Gradient at " + value + ": " + gradient);
        assertEquals(-1, Math.signum(gradient), 0);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        BetaVertex vertexUnderTest = new BetaVertex(
                new UniformVertex(0.0, 1.0),
                3.0
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void sampleMatchesLogProb() {
        BetaVertex betaVertex = new BetaVertex(2.0, 2.0);

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
                betaVertex,
                1000000,
                0.1,
                0.9,
                .05,
                0.01,
                random
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        BetaVertex beta = new BetaVertex(uniformA, 3.0);

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
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        BetaVertex beta = new BetaVertex(1.0, uniformA);

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
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0));
        latentAlphaBeta.add(new SmoothUniformVertex(0.01, 10.0));

        VertexVariationalMAP.inferHyperParamsFromSamples(
                hyperParams -> new BetaVertex(hyperParams.get(0), hyperParams.get(1)),
                alphaBeta,
                latentAlphaBeta,
                1000,
                random
        );
    }
}
