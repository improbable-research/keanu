package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class GaussianVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        GaussianVertex tensorGaussianVertex = new GaussianVertex(0, 1);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex prior = new ExponentialVertex(1.);
        DoubleVertex mu = prior.plus(1.);
        DoubleVertex sigma = ConstantVertex.of(1.);
        GaussianVertex vertex = new GaussianVertex(mu, sigma);

        LogProbGraph logProbGraph = vertex.logProbGraph();
        prior.setValue(-1.);

        NormalDistribution distribution = new NormalDistribution(0., 1.);
        double expectedLogDensity = distribution.logDensity(0.5);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, DoubleTensor.scalar(0.5), expectedLogDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(-0.75);
        GaussianVertex tensorGaussianVertex = new GaussianVertex(0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{0.25, -0.75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex prior = new ExponentialVertex(1.);
        DoubleVertex mu = prior.plus(1.);
        DoubleVertex sigma = ConstantVertex.of(1., 1.);
        GaussianVertex vertex = new GaussianVertex(mu, sigma);

        prior.setValue(-1.);
        LogProbGraph logProbGraph = vertex.logProbGraph();

        NormalDistribution distribution = new NormalDistribution(0., 1.);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(-0.75);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, DoubleTensor.create(0.25, -0.75), expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Gaussian.Diff gaussianLogDiff = Gaussian.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        GaussianVertex tensorGaussianVertex = new GaussianVertex(muTensor, sigmaTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorGaussianVertex.dLogPdf(0.5, muTensor, sigmaTensor, tensorGaussianVertex);

        assertEquals(gaussianLogDiff.dPdmu, actualDerivatives.get(muTensor).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdsigma, actualDerivatives.get(sigmaTensor).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdx, actualDerivatives.get(tensorGaussianVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, -0.75, 0.1, -2, 1.3};

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        Supplier<GaussianVertex> vertexSupplier = () -> new GaussianVertex(muTensor, sigmaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        GaussianVertex vertexUnderTest = new GaussianVertex(
            mu,
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        GaussianVertex gaussian = new GaussianVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(1.5),
            0.1,
            uniformA,
            gaussian,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        GaussianVertex gaussian = new GaussianVertex(3.0, uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
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
        GaussianVertex vertex = new GaussianVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(0.0),
            ConstantVertex.of(2.0)
        );

        double from = -4;
        double to = 4;
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
            hyperParams -> new GaussianVertex(new long[]{numSamples}, hyperParams.get(0), hyperParams.get(1)),
            muSigma,
            latentMuSigma,
            random
        );
    }
}
