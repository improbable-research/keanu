package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.LogisticDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.distributions.gradient.Logistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LogisticVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        LogisticVertex tensorLogisticVertex = VertexOfType.logistic(0.5, 1.5);
        LogisticDistribution distribution = new LogisticDistribution(0.5, 1.5);
        double expectedDensity = distribution.logDensity(2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogisticVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        LogisticDistribution distribution = new LogisticDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogisticVertex ndLogisticVertex = VertexOfType.logistic(0.0, 1.0);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLogisticVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Logistic.Diff logisticLogDiff = Logistic.dlnPdf(0.0, 0.5, 1.5);

        UniformVertex aTensor = VertexOfType.uniform(0.0, 5.0);
        aTensor.setValue(0.0);

        UniformVertex bTensor = VertexOfType.uniform(0.0, 5.0);
        bTensor.setValue(0.5);

        LogisticVertex tensorLogisticVertex = VertexOfType.logistic(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorLogisticVertex.dLogProb(DoubleTensor.scalar(1.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(logisticLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdx, actual.withRespectTo(tensorLogisticVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = VertexOfType.uniform(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        LogisticVertex vertexUnderTest = VertexOfType.logistic(
            mu,
            ConstantVertex.of(3.0)
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = VertexOfType.uniform(0.0, 1.0);
        LogisticVertex logistic = VertexOfType.logistic(uniformA, ConstantVertex.of(1.0));

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.0),
            Nd4jDoubleTensor.scalar(0.9),
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
        UniformVertex uniformA = VertexOfType.uniform(0., 1.);
        LogisticVertex logistic = VertexOfType.logistic(ConstantVertex.of(0.0), uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(1.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.5),
            Nd4jDoubleTensor.scalar(3.5),
            0.1,
            uniformA,
            logistic,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void logisticSampleMethodMatchesLogProbMethod() {

        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        LogisticVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.MU, 0.0)
            .withInput(ParameterName.S, 0.5)
            .logistic();

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
        UniformVertex latentB = VertexOfType.uniform(0.01, 10.0);
        latentB.setAndCascade(0.1);
        latentAB.add(ConstantVertex.of(trueA));
        latentAB.add(latentB);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new DistributionVertexBuilder()
                .shaped(numSamples, 1)
                .withInput(ParameterName.MU, hyperParams.get(0))
                .withInput(ParameterName.S, hyperParams.get(1))
                .logistic(),
            aB,
            latentAB,
            random
        );
    }

}
