package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

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

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 1e-2;

        double a = 0.5;
        double theta = 0.5;
        double k = 6.0;

        GammaVertex g = new GammaVertex(a, theta, k, random);

        double mean = k * theta + a;
        double standardDeviation = Math.sqrt(k * Math.pow(theta, 2));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(N, g, mean, standardDeviation, epsilon);
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
    public void dDensityMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex a = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        GammaVertex g = new GammaVertex(a, new ConstantDoubleVertex(0.5), new ConstantDoubleVertex(1.0), random);

        double vertexStartValue = 1.0;
        double vertexEndValue = 1.5;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
                1.0,
                0.1,
                a,
                g,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdtheta() {
        UniformVertex t = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        GammaVertex g = new GammaVertex(new ConstantDoubleVertex(0.5), t, new ConstantDoubleVertex(1.0), random);

        double vertexStartValue = 1.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.1,
                1.0,
                0.1,
                t,
                g,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdk() {
        UniformVertex k = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        GammaVertex g = new GammaVertex(new ConstantDoubleVertex(0.5), new ConstantDoubleVertex(1.0), k, random);

        double vertexStartValue = 1.0;
        double vertexEndValue = 1.5;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.5,
                2.5,
                0.1,
                k,
                g,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        GammaVertex g = new GammaVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(2.0),
                new ConstantDoubleVertex(1.0),
                random
        );

        double atValue = 0.5;
        g.setAndCascade(atValue);

        Map<String, Double> dP = g.dDensityAtValue();
        Map<String, Double> dlnP = g.dlnDensityAtValue();

        final double density = g.densityAtValue();
        for (String vertexId : dP.keySet()) {
            dP.put(vertexId, dP.get(vertexId) / density);
        }

        assertEquals(dP.get(g.getId()), dlnP.get(g.getId()), 0.01);
    }

    private void testPdfAtPercentiles(double theta, double k) {
        GammaVertex g = new GammaVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(theta),
                new ConstantDoubleVertex(k),
                random
        );

        GammaDistribution apache = new GammaDistribution(k, theta);
        log.info("k = " + k + ", theta = " + theta + ":");

        for (double x = 0.0; x <= 1.0; x += 0.1) {
            double expected = apache.density(x);
            double density = g.density(x);
            log.info("   Density at " + x + " = " + density + " (expected = " + expected + ")");
            assertEquals(expected, density, 0.0001);
        }
    }

    private void testdPdxAtPercentiles(double theta, double k) {
        GammaVertex g = new GammaVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(theta),
                new ConstantDoubleVertex(k),
                random
        );

        log.info("k = " + k + ", theta = " + theta + ":");

        for (double x = 0.01; x <= 1.0; x += 0.1) {
            double approxExpected = (g.density(x + DELTA) - g.density(x - DELTA)) / (2 * DELTA);
            g.setValue(x);
            double actual = g.dDensityAtValue().get(g.getId());
            log.info("   Gradient at " + x + " = " + actual + " (approx expected = " + approxExpected + ")");
            assertEquals(approxExpected, actual, 0.1);
        }
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        GammaVertex g = new GammaVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(2.0),
                new ConstantDoubleVertex(1.0),
                random
        );

        double atValue = 0.5;
        double logOfDensity = Math.log(g.density(atValue));
        double logDensity = g.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void samplingMatchesPdf() {
        GammaVertex gamma = new GammaVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(2.0),
                new ConstantDoubleVertex(3.0),
                random
        );

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(
                gamma,
                100000,
                2.0,
                10.0,
                0.1,
                0.01);
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
        latentAThetaK.add(new SmoothUniformVertex(0.01, 10.0, random));
        latentAThetaK.add(new SmoothUniformVertex(0.01, 10.0, random));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new GammaVertex(hyperParams.get(0), hyperParams.get(1), hyperParams.get(2), random),
                aThetaK,
                latentAThetaK,
                2000
        );
    }

}
