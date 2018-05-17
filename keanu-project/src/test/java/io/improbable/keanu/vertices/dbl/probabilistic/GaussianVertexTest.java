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
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GaussianVertexTest {
    private final Logger log = LoggerFactory.getLogger(GaussianVertexTest.class);

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
        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
            N,
            gaussianVertex,
            0.0,
            1.0,
            epsilon,
            random
        );
    }

    @Test
    public void gradientAtMuIsZero() {
        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        gaussianVertex.setValue(0.0);
        double gradient = gaussianVertex.dLogProbAtValue().get(gaussianVertex.getId()).scalar();
        log.info("Gradient at mu: " + gradient);
        assertEquals(0, gradient, 0);
    }

    @Test
    public void gradientBeforeMuIsPositive() {
        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        gaussianVertex.setValue(-1.0);
        double gradient = gaussianVertex.dLogProbAtValue().get(gaussianVertex.getId()).scalar();
        log.info("Gradient after mu: " + gradient);
        assertTrue(gradient > 0);
    }

    @Test
    public void gradientAfterMuIsNegative() {
        GaussianVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        gaussianVertex.setValue(1.0);
        double gradient = gaussianVertex.dLogProbAtValue().get(gaussianVertex.getId()).scalar();
        log.info("Gradient after mu: " + gradient);
        assertTrue(gradient < 0);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        GaussianVertex vertexUnderTest = new GaussianVertex(
            new UniformVertex(0.0, 1.0),
            3.0
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        Vertex<Double> vertex = new GaussianVertex(0.0, 2.0);

        double from = -4;
        double to = 4;
        double bucketSize = 0.05;
        long sampleCount = 1000000;

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
            vertex,
            sampleCount,
            from,
            to,
            bucketSize,
            1e-2,
            random
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        GaussianVertex gaussian = new GaussianVertex(uniformA, 3.0);

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
            DELTA
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        GaussianVertex gaussian = new GaussianVertex(3.0, uniformA);

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
            DELTA
        );
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 0;
        double trueSigma = 2.0;

        List<DoubleVertex> muSigma = new ArrayList<>();
        muSigma.add(new ConstantDoubleVertex(trueMu));
        muSigma.add(new ConstantDoubleVertex(trueSigma));

        SmoothUniformVertex latentMu = new SmoothUniformVertex(-10.0, 10.0);
        latentMu.setValue(latentMu.sample(random));
        SmoothUniformVertex latentSigma = new SmoothUniformVertex(1.0, 10.0);
        latentSigma.setValue(latentSigma.sample(random));

        List<DoubleVertex> latentMuSigma = new ArrayList<>();
        latentMuSigma.add(latentMu);
        latentMuSigma.add(latentSigma);

        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new GaussianVertex(hyperParams.get(0), hyperParams.get(1)),
            muSigma,
            latentMuSigma,
            2000,
            random
        );
    }
}
