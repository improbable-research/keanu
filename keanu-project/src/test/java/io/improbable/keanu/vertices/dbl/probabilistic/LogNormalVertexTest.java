package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.gradient.LogNormal;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class LogNormalVertexTest {
    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        LogNormalDistribution distribution = new LogNormalDistribution(0, 1);
        LogNormalVertex tensorLogNormalVertex = new LogNormalVertex(0, 1);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogNormalVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex mu = ConstantVertex.of(0.);
        DoubleVertex sigma = ConstantVertex.of(1.);
        LogNormalVertex logNormalVertex = new LogNormalVertex(mu, sigma);
        LogProbGraph logProbGraph = logNormalVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, logNormalVertex, DoubleTensor.scalar(0.5));

        LogNormalDistribution distribution = new LogNormalDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(0.5);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        LogNormalDistribution distribution = new LogNormalDistribution(0, 1);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogNormalVertex tensorLogNormalVertex = new LogNormalVertex(0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorLogNormalVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(0., 0.);
        DoubleVertex sigma = ConstantVertex.of(1., 1.);
        LogNormalVertex logNormalVertex = new LogNormalVertex(mu, sigma);
        LogProbGraph logProbGraph = logNormalVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, logNormalVertex, DoubleTensor.create(0.25, 0.75));

        LogNormalDistribution distribution = new LogNormalDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        LogNormal.Diff logNormalLogDiff = LogNormal.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        LogNormalVertex tensorLogNormalVertex = new LogNormalVertex(muTensor, sigmaTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorLogNormalVertex.dLogPdf(0.5, muTensor, sigmaTensor, tensorLogNormalVertex);

        assertEquals(logNormalLogDiff.dPdmu, actualDerivatives.get(muTensor).scalar(), 1e-5);
        assertEquals(logNormalLogDiff.dPdsigma, actualDerivatives.get(sigmaTensor).scalar(), 1e-5);
        assertEquals(logNormalLogDiff.dPdx, actualDerivatives.get(tensorLogNormalVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, -0.75, 0.1, -2, 1.3};

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        Supplier<LogNormalVertex> vertexSupplier = () -> new LogNormalVertex(muTensor, sigmaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(0.5);
        LogNormalVertex vertexUnderTest = new LogNormalVertex(mu, 3.0);
        vertexUnderTest.setAndCascade(1.0);
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        LogNormalVertex logNormal = new LogNormalVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(1.5),
            0.1,
            uniformA,
            logNormal,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        LogNormalVertex logNormal = new LogNormalVertex(3.0, uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(1.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            logNormal,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void logNormalSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        LogNormalVertex vertex = new LogNormalVertex(new long[]{sampleCount, 1}, 0.0, 2.0);

        double from = 0.1;
        double to = 8;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 4.5;
        double trueSigma = 2.0;

        List<DoubleVertex> muSigma = new ArrayList<>();
        muSigma.add(ConstantVertex.of(trueMu));
        muSigma.add(ConstantVertex.of(trueSigma));

        List<DoubleVertex> latentMuSigma = new ArrayList<>();
        UniformVertex latentMu = new UniformVertex(0.01, 10.0);
        latentMu.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentSigma = new UniformVertex(0.01, 10.0);
        latentSigma.setAndCascade(DoubleTensor.scalar(0.1));
        latentMuSigma.add(latentMu);
        latentMuSigma.add(latentSigma);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new LogNormalVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            muSigma,
            latentMuSigma,
            random
        );
    }
}
