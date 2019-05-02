package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.gradient.Gamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class GammaVertexTest {

    private KeanuRandom random;

    private static final double DELTA = 0.0001;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        GammaDistribution distribution = new GammaDistribution(1.0, 1.5);
        GammaVertex tensorGamma = new GammaVertex(1.5, 1.0);
        double expectedDensity = distribution.logDensity(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGamma, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex theta = ConstantVertex.of(1.5);
        DoubleVertex k = ConstantVertex.of(1.);
        GammaVertex tensorGamma = new GammaVertex(theta, k);
        LogProbGraph logProbGraph = tensorGamma.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, theta, theta.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, k, k.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, tensorGamma, DoubleTensor.scalar(0.5));

        GammaDistribution distribution = new GammaDistribution(1., 1.5);
        double expectedDensity = distribution.logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        GammaDistribution distribution = new GammaDistribution(5.0, 1.0);
        double expectedLogDensity = distribution.logDensity(1) + distribution.logDensity(3);
        GammaVertex tensorGamma = new GammaVertex(1., 5.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGamma, new double[]{1., 3.}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex theta = ConstantVertex.of(1.0, 1.0);
        DoubleVertex k = ConstantVertex.of(5., 5.);
        GammaVertex tensorGamma = new GammaVertex(theta, k);
        LogProbGraph logProbGraph = tensorGamma.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, theta, theta.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, k, k.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, tensorGamma, DoubleTensor.create(1., 3.));

        GammaDistribution distribution = new GammaDistribution(5., 1.);
        double expectedDensity = distribution.logDensity(1.) + distribution.logDensity(3.);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        Gamma.Diff gammaLogDiff = Gamma.dlnPdf(2, 5.5, 1.5);

        UniformVertex thetaTensor = new UniformVertex(0.5, 1.0);
        thetaTensor.setValue(DoubleTensor.scalar(2));

        UniformVertex kTensor = new UniformVertex(1.0, 5.0);
        kTensor.setValue(DoubleTensor.scalar(5.5));

        GammaVertex tensorGamma = new GammaVertex(thetaTensor, kTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorGamma.dLogPdf(DoubleTensor.scalar(1.5), thetaTensor, kTensor, tensorGamma);

        assertEquals(gammaLogDiff.dPdtheta, actualDerivatives.get(thetaTensor).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdk, actualDerivatives.get(kTensor).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdx, actualDerivatives.get(tensorGamma).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{1.5, 2, 2.5, 3, 3.5};

        UniformVertex thetaTensor = new UniformVertex(0.5, 1.0);
        thetaTensor.setValue(DoubleTensor.scalar(0.75));

        UniformVertex kTensor = new UniformVertex(1.0, 5.0);
        kTensor.setValue(DoubleTensor.scalar(2.5));

        Supplier<GammaVertex> vertexSupplier = () -> new GammaVertex(thetaTensor, kTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        GammaVertex vertexUnderTest = new GammaVertex(1.5, 5.0);
        vertexUnderTest.setAndCascade(DoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdtheta() {

        UniformVertex uniformA = new UniformVertex(1.0, 3.0);
        GammaVertex gamma = new GammaVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(3.);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(2.5),
            0.1,
            uniformA,
            gamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdk() {

        UniformVertex uniformA = new UniformVertex(2.0, 5.0);
        GammaVertex gamma = new GammaVertex(2.0, uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(3.);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(2.0),
            DoubleTensor.scalar(4.5),
            0.1,
            uniformA,
            gamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void gammaSampledMethodMatchesLogProbMethod() {
        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        GammaVertex vertex = new GammaVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(2.0),
            ConstantVertex.of(7.5)
        );

        double from = 1.5;
        double to = 2.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Category(Slow.class)
    @Test
    public void inferHyperParamsFromSamples() {

        double trueTheta = 2.0;
        double trueK = 3.0;

        DoubleVertex constTheta = ConstantVertex.of(trueTheta);
        DoubleVertex constK = ConstantVertex.of(trueK);

        List<DoubleVertex> thetaK = new ArrayList<>();
        thetaK.add(constTheta);
        thetaK.add(constK);

        List<DoubleVertex> latentThetaK = new ArrayList<>();
        UniformVertex latentTheta = new UniformVertex(0.01, 10.0);
        latentTheta.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentK = new UniformVertex(0.01, 10.0);
        latentK.setAndCascade(DoubleTensor.scalar(0.1));

        latentThetaK.add(latentTheta);
        latentThetaK.add(latentK);

        int numSamples = 5000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new GammaVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            thetaK,
            latentThetaK,
            random
        );

    }

}
