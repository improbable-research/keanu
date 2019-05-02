package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.gradient.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class HalfGaussianVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        double expectedDensity = distribution.logDensity(0.5) + Math.log(2);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex sigma = ConstantVertex.of(1.);
        HalfGaussianVertex halfGaussianVertex = new HalfGaussianVertex(sigma);
        LogProbGraph logProbGraph = halfGaussianVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, halfGaussianVertex, DoubleTensor.scalar(0.5));

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedDensity = distribution.logDensity(0.5) + Math.log(2);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfNegativeScalar() {

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, -0.5, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfNegativeScalar() {
        DoubleVertex sigma = ConstantVertex.of(1.);
        HalfGaussianVertex halfGaussianVertex = new HalfGaussianVertex(sigma);
        LogProbGraph logProbGraph = halfGaussianVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, halfGaussianVertex, DoubleTensor.scalar(-0.5));

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75) + 2 * Math.log(2);
        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex sigma = ConstantVertex.of(1., 1.);
        HalfGaussianVertex halfGaussianVertex = new HalfGaussianVertex(sigma);
        LogProbGraph logProbGraph = halfGaussianVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, halfGaussianVertex, DoubleTensor.create(0.25, 0.75));

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75) + 2. * Math.log(2);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfNegativeVector() {

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{-0.25, 0.75}, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfNegativeVector() {
        DoubleVertex sigma = ConstantVertex.of(1., 1.);
        HalfGaussianVertex halfGaussianVertex = new HalfGaussianVertex(sigma);
        LogProbGraph logProbGraph = halfGaussianVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, sigma, sigma.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, halfGaussianVertex, DoubleTensor.create(-0.25, 0.75));

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Gaussian.Diff gaussianLogDiff = Gaussian.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(sigmaTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorGaussianVertex.dLogPdf(0.5, sigmaTensor, tensorGaussianVertex);

        assertEquals(gaussianLogDiff.dPdsigma, actualDerivatives.get(sigmaTensor).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdx, actualDerivatives.get(tensorGaussianVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 2, 1.3};

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        Supplier<HalfGaussianVertex> vertexSupplier = () -> new HalfGaussianVertex(sigmaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        HalfGaussianVertex vertexUnderTest = new HalfGaussianVertex(
            3.0
        );
        vertexUnderTest.setAndCascade(DoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        HalfGaussianVertex gaussian = new HalfGaussianVertex(uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(0.0),
            0.1,
            uniformA,
            gaussian,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        HalfGaussianVertex vertex = new HalfGaussianVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(2.0)
        );

        double from = 0;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueSigma = 2.0;

        List<DoubleVertex> sigma = new ArrayList<>();
        sigma.add(ConstantVertex.of(trueSigma));

        List<DoubleVertex> latentSigmaList = new ArrayList<>();
        UniformVertex latentSigma = new UniformVertex(0.01, 10.0);
        latentSigma.setAndCascade(DoubleTensor.scalar(0.1));
        latentSigmaList.add(latentSigma);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new HalfGaussianVertex(new long[]{numSamples, 1}, hyperParams.get(0)),
            sigma,
            latentSigmaList,
            random
        );
    }

    @Test
    public void outOfBoundsGradientCalculation() {
        HalfGaussianVertex gaussianVertex = new HalfGaussianVertex(new long[]{2}, 5);
        DoubleTensor value = DoubleTensor.create(-5.0, 5.0);
        Map<Vertex, DoubleTensor> actualDerivatives = gaussianVertex.dLogPdf(value, gaussianVertex);
        DoubleTensor derivative = actualDerivatives.get(gaussianVertex);
        Assert.assertEquals(0.0, derivative.getValue(0), 1e-6);
        Assert.assertTrue(derivative.getValue(1) != 0.);
    }
}
