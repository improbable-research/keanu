package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.gradient.Logistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.apache.commons.math3.distribution.LogisticDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class LogisticVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {
        LogisticVertex tensorLogisticVertex = new LogisticVertex(0.5, 1.5);
        LogisticDistribution distribution = new LogisticDistribution(0.5, 1.5);
        double expectedDensity = distribution.logDensity(2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogisticVertex, 2.0, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex mu = ConstantVertex.of(0.5);
        DoubleVertex s = ConstantVertex.of(1.5);
        LogisticVertex logisticVertex = new LogisticVertex(mu, s);
        LogProbGraph logProbGraph = logisticVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, s, s.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, logisticVertex, DoubleTensor.scalar(2.));

        LogisticDistribution distribution = new LogisticDistribution(0.5, 1.5);
        double expectedDensity = distribution.logDensity(2.);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {
        LogisticDistribution distribution = new LogisticDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogisticVertex ndLogisticVertex = new LogisticVertex(0.0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLogisticVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(0., 0.);
        DoubleVertex s = ConstantVertex.of(1., 1.);
        LogisticVertex logisticVertex = new LogisticVertex(mu, s);
        LogProbGraph logProbGraph = logisticVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, s, s.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, logisticVertex, DoubleTensor.create(.25, .75));

        LogisticDistribution distribution = new LogisticDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(.25) + distribution.logDensity(.75);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Logistic.Diff logisticLogDiff = Logistic.dlnPdf(0.0, 0.5, 1.5);

        UniformVertex aTensor = new UniformVertex(0.0, 5.0);
        aTensor.setValue(0.0);

        UniformVertex bTensor = new UniformVertex(0.0, 5.0);
        bTensor.setValue(0.5);

        LogisticVertex tensorLogisticVertex = new LogisticVertex(aTensor, bTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorLogisticVertex.dLogPdf(1.5, aTensor, bTensor, tensorLogisticVertex);

        assertEquals(logisticLogDiff.dPda, actualDerivatives.get(aTensor).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdb, actualDerivatives.get(bTensor).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdx, actualDerivatives.get(tensorLogisticVertex).scalar(), 1e-5);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(DoubleTensor.scalar(0.5));
        LogisticVertex vertexUnderTest = new LogisticVertex(
            mu,
            3.
        );
        vertexUnderTest.setAndCascade(DoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(0.0, 1.0);
        LogisticVertex logistic = new LogisticVertex(uniformA, 1.0);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(1.0);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(0.0),
            DoubleTensor.scalar(0.9),
            0.1,
            uniformA,
            logistic,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = new UniformVertex(0., 1.);
        LogisticVertex logistic = new LogisticVertex(0.0, uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(1.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.scalar(0.5),
            DoubleTensor.scalar(3.5),
            0.1,
            uniformA,
            logistic,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void logisticSampleMethodMatchesLogProbMethod() {

        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        LogisticVertex vertex = new LogisticVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(0.0),
            ConstantVertex.of(0.5)
        );

        double from = 0.5;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueB = 0.5;

        List<DoubleVertex> aB = new ArrayList<>();
        aB.add(ConstantVertex.of(trueA));
        aB.add(ConstantVertex.of(trueB));

        List<DoubleVertex> latentAB = new ArrayList<>();
        UniformVertex latentB = new UniformVertex(0.01, 10.0);
        latentB.setAndCascade(0.1);
        latentAB.add(ConstantVertex.of(trueA));
        latentAB.add(latentB);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new LogisticVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            aB,
            latentAB,
            random
        );
    }
}
