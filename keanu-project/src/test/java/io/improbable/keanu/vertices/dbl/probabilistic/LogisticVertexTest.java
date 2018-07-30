package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Logistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.math3.distribution.LogisticDistribution;
import org.junit.Before;
import org.junit.Test;

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
    public void matchesKnownLogDensityOfScalar() {
        LogisticVertex tensorLogisticVertex = new LogisticVertex(0.5, 1.5);
        LogisticDistribution distribution = new LogisticDistribution(0.5, 1.5);
        double expectedDensity = distribution.logDensity(2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogisticVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        LogisticDistribution distribution = new LogisticDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogisticVertex ndLogisticVertex = new LogisticVertex(0.0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLogisticVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Logistic.Diff logisticLogDiff = Logistic.dlnPdf(0.0, 0.5, 1.5);

        UniformVertex aTensor = new UniformVertex(0.0, 5.0);
        aTensor.setValue(0.0);

        UniformVertex bTensor = new UniformVertex(0.0, 5.0);
        bTensor.setValue(0.5);

        LogisticVertex tensorLogisticVertex = new LogisticVertex(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorLogisticVertex.dLogPdf(1.5);

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(logisticLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdx, actual.withRespectTo(tensorLogisticVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        LogisticVertex vertexUnderTest = new LogisticVertex(
            mu,
            3.
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(0.0, 1.0);
        LogisticVertex logistic = new LogisticVertex(uniformA, 1.0);

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
        UniformVertex uniformA = new UniformVertex(0., 1.);
        LogisticVertex logistic = new LogisticVertex(0.0, uniformA);

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
        LogisticVertex vertex = new LogisticVertex(
            new int[]{sampleCount, 1},
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
            hyperParams -> new LogisticVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            aB,
            latentAB,
            random
        );
    }

}
