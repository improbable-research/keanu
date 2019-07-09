package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.gradient.Laplace;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class LaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        LaplaceVertex tensorLaplaceVertex = new LaplaceVertex(0.5, 1);
        LaplaceDistribution distribution = new LaplaceDistribution(0.5, 1.0);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLaplaceVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex mu = ConstantVertex.of(0.5);
        DoubleVertex beta = ConstantVertex.of(1.);
        LaplaceVertex laplaceVertex = new LaplaceVertex(mu, beta);
        LogProbGraph logProbGraph = laplaceVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, beta, beta.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, laplaceVertex, DoubleTensor.scalar(0.5));

        LaplaceDistribution distribution = new LaplaceDistribution(0.5, 1.);
        double expectedDensity = distribution.logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        LaplaceDistribution distribution = new LaplaceDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LaplaceVertex ndLaplaceVertex = new LaplaceVertex(0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLaplaceVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(0., 0.);
        DoubleVertex beta = ConstantVertex.of(1., 1.);
        LaplaceVertex laplaceVertex = new LaplaceVertex(mu, beta);
        LogProbGraph logProbGraph = laplaceVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, beta, beta.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, laplaceVertex, DoubleTensor.create(0.25, 0.75));

        LaplaceDistribution distribution = new LaplaceDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Laplace.Diff laplaceLogDiff = Laplace.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex betaTensor = new UniformVertex(0.0, 1.0);
        betaTensor.setValue(1.0);

        LaplaceVertex tensorLaplaceVertex = new LaplaceVertex(muTensor, betaTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorLaplaceVertex.dLogPdf(0.5, muTensor, betaTensor, tensorLaplaceVertex);

        assertEquals(laplaceLogDiff.dPdmu, actualDerivatives.get(muTensor).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdbeta, actualDerivatives.get(betaTensor).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdx, actualDerivatives.get(tensorLaplaceVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 22, 1.3};

        UniformVertex muTensor = new UniformVertex(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex betaTensor = new UniformVertex(0.0, 1.0);
        betaTensor.setValue(1.0);

        Supplier<LaplaceVertex> vertexSupplier = () -> new LaplaceVertex(muTensor, betaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(DoubleTensor.scalar(0.5));
        LaplaceVertex vertexUnderTest = new LaplaceVertex(
            mu,
            3.0
        );
        vertexUnderTest.setAndCascade(DoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        LaplaceVertex laplace = new LaplaceVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(1.5),
            0.1,
            uniformA,
            laplace,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdbeta() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        LaplaceVertex laplace = new LaplaceVertex(3.0, uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(1.0),
            DoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            laplace,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void laplaceSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        LaplaceVertex vertex = new LaplaceVertex(
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
        double trueBeta = 2.0;

        List<DoubleVertex> muBeta = new ArrayList<>();
        muBeta.add(ConstantVertex.of(trueMu));
        muBeta.add(ConstantVertex.of(trueBeta));

        List<DoubleVertex> latentMuBeta = new ArrayList<>();
        UniformVertex latentMu = new UniformVertex(0.01, 10.0);
        latentMu.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentBeta = new UniformVertex(0.01, 10.0);
        latentBeta.setAndCascade(DoubleTensor.scalar(0.1));
        latentMuBeta.add(latentMu);
        latentMuBeta.add(latentBeta);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new LaplaceVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            muBeta,
            latentMuBeta,
            random
        );
    }
}
