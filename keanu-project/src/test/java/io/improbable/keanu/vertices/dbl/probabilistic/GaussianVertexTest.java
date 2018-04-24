package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.Vertex;
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
import static org.junit.Assert.assertTrue;

public class GaussianVertexTest {
    private final Logger log = LoggerFactory.getLogger(GaussianVertexTest.class);

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
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(N, g, 0.0, 1.0, epsilon);
    }

    @Test
    public void gradientAtMuIsZero() {
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        g.setValue(0.0);
        double gradient = g.dDensityAtValue().get(g.getId());
        log.info("Gradient at mu: " + gradient);
        assertEquals(0, gradient, 0);
    }

    @Test
    public void gradientBeforeMuIsPositive() {
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        g.setValue(-1.0);
        double gradient = g.dDensityAtValue().get(g.getId());
        log.info("Gradient after mu: " + gradient);
        assertTrue(gradient > 0);
    }

    @Test
    public void gradientAfterMuIsNegative() {
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        g.setValue(1.0);
        double gradient = g.dDensityAtValue().get(g.getId());
        log.info("Gradient after mu: " + gradient);
        assertTrue(gradient < 0);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(2.0), random);
        double atValue = 0.5;
        double logOfDensity = Math.log(g.density(atValue));
        double logDensity = g.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        GaussianVertex g = new GaussianVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), random);
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(g, 0.5, 0.001);
    }

    @Test
    public void gaussianSampleMethodMatchesDensityMethod() {

        Random random = new Random(1);

        Vertex<Double> vertex = new GaussianVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(2.0),
                random
        );

        double from = -4;
        double to = 4;
        double bucketSize = 0.05;
        long sampleCount = 1000000;

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(vertex, sampleCount, from, to, bucketSize, 1e-2);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0), random);
        GaussianVertex gaussian = new GaussianVertex(uniformA, new ConstantDoubleVertex(3.0), random);

        double vertexStartValue = 0.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                1.5,
                0.1,
                uniformA,
                gaussian,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(1.5), new ConstantDoubleVertex(3.0), random);
        GaussianVertex gaussian = new GaussianVertex(new ConstantDoubleVertex(3.0), uniformA, random);

        double vertexStartValue = 0.0;
        double vertexEndValue = 0.5;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                3.0,
                0.1,
                uniformA,
                gaussian,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 0;
        double trueSigma = 2.0;

        List<DoubleVertex> muSigma = new ArrayList<>();
        muSigma.add(new ConstantDoubleVertex(trueMu));
        muSigma.add(new ConstantDoubleVertex(trueSigma));

        List<DoubleVertex> latentMuSigma = new ArrayList<>();
        latentMuSigma.add(new SmoothUniformVertex(0.01, 10.0, random));
        latentMuSigma.add(new SmoothUniformVertex(0.01, 10.0, random));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new GaussianVertex(hyperParams.get(0), hyperParams.get(1), random),
                muSigma,
                latentMuSigma,
                1000
        );
    }
}
