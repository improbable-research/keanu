package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;

public class GammaVertexTest {
    private final Logger log = LoggerFactory.getLogger(GammaVertexTest.class);

    private static final double DELTA = 0.0001;

    private static final double[][] TEST_VALUES = new double[][]{
        {2.0, 1.0},
        {2.0, 2.0},
        {2.0, 3.0},
        {1.0, 5.0},
        {0.5, 9.0},
        {1.0, 7.5},
        {1.5, 7.5},
        {1.0, 0.5}
    };

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 1e-2;

        double a = 0.5;
        double theta = 0.5;
        double k = 6.0;

        GammaVertex gammaVertex = new GammaVertex(a, theta, k);

        double mean = k * theta + a;
        double standardDeviation = Math.sqrt(k * Math.pow(theta, 2));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
            N,
            gammaVertex,
            mean,
            standardDeviation,
            epsilon,
            random
        );
    }

    @Test
    public void pdfMatchesApacheMathGammaDistribution() {
        for (int i = 0; i < TEST_VALUES.length; i++) {
            testPdfAtPercentiles(TEST_VALUES[i][0], TEST_VALUES[i][1]);
        }
    }

    @Test
    public void dPdxMatchesApproxGradient() {
        for (int i = 0; i < TEST_VALUES.length; i++) {
            testdPdxAtPercentiles(TEST_VALUES[i][0], TEST_VALUES[i][1]);
        }
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformVertex = new UniformVertex(0.0, 1.0);
        GammaVertex gammaVertex = new GammaVertex(uniformVertex, new ConstantDoubleVertex(0.5), new ConstantDoubleVertex(1.0));

        double vertexStartValue = 1.0;
        double vertexEndValue = 1.5;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
            1.0,
            0.1,
            uniformVertex,
            gammaVertex,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdtheta() {
        UniformVertex t = new UniformVertex(0.0, 1.0);
        GammaVertex gammaVertex = new GammaVertex(new ConstantDoubleVertex(0.5), t, new ConstantDoubleVertex(1.0));

        double vertexStartValue = 1.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.1,
            1.0,
            0.1,
            t,
            gammaVertex,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdk() {
        UniformVertex k = new UniformVertex(0.0, 1.0);
        GammaVertex gammaVertex = new GammaVertex(new ConstantDoubleVertex(0.5), new ConstantDoubleVertex(1.0), k);

        double vertexStartValue = 1.0;
        double vertexEndValue = 1.5;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.5,
            2.5,
            0.1,
            k,
            gammaVertex,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    private void testPdfAtPercentiles(double theta, double k) {
        GammaVertex gammaVertex = new GammaVertex(
            0.0,
            theta,
            k
        );

        GammaDistribution apache = new GammaDistribution(k, theta);
        log.info("k = " + k + ", theta = " + theta + ":");

        for (double x = 0.1; x <= 1.0; x += 0.1) {
            double expected = Math.log(apache.density(x));
            double density = gammaVertex.logProb(x);
            assertThat("   Density at " + x + " = " + density + " (expected = " + expected + ")",
                expected, closeTo(density, 0.0001)
            );
        }
    }

    private void testdPdxAtPercentiles(double theta, double k) {
        GammaVertex gammaVertex = new GammaVertex(
            0.0,
            theta,
            k
        );

        log.info("k = " + k + ", theta = " + theta + ":");

        for (double x = 0.01; x <= 1.0; x += 0.1) {
            double approxExpected = (gammaVertex.logProb(x + DELTA) - gammaVertex.logProb(x - DELTA)) / (2 * DELTA);
            gammaVertex.setValue(x);
            double actual = gammaVertex.dLogProbAtValue().get(gammaVertex.getId()).scalar();
            assertThat("   Gradient at " + x + " = " + actual + " (approx expected = " + approxExpected + ")",
                approxExpected, closeTo(actual, 0.1)
            );
        }
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        GammaVertex vertexUnderTest = new GammaVertex(
            new UniformVertex(0.0, 1.0),
            new ConstantDoubleVertex(.0),
            new ConstantDoubleVertex(1.0)
        );

        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void samplingMatchesLogProb() {
        GammaVertex gammaVertex = new GammaVertex(
            0.0,
            2.0,
            3.0
        );

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
            gammaVertex,
            100000,
            2.0,
            10.0,
            0.1,
            0.01,
            random
        );
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueTheta = 3.0;
        double trueK = 2.0;

        DoubleVertex a = new ConstantDoubleVertex(trueA);

        List<DoubleVertex> aThetaK = new ArrayList<>();
        aThetaK.add(a);
        aThetaK.add(new ConstantDoubleVertex(trueTheta));
        aThetaK.add(new ConstantDoubleVertex(trueK));

        List<DoubleVertex> latentAThetaK = new ArrayList<>();
        latentAThetaK.add(a);
        latentAThetaK.add(new SmoothUniformVertex(0.01, 10.0));
        latentAThetaK.add(new SmoothUniformVertex(0.01, 10.0));

        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new GammaVertex(hyperParams.get(0), hyperParams.get(1), hyperParams.get(2)),
            aThetaK,
            latentAThetaK,
            2000,
            random
        );
    }

}
