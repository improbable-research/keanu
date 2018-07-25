package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.continuous.DistributionOfType;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class ExponentialVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void maskAppliedToValuesWhereXLessThanA() {
        DoubleTensor matrixA = Nd4jDoubleTensor.create(new double[]{1, 4}, new int[]{2, 1});
        DoubleTensor matrixX = Nd4jDoubleTensor.create(new double[]{2, 2}, new int[]{2, 1});

        DoubleTensor maskResult = DistributionOfType.exponential(matrixA, new ScalarDoubleTensor(1.0)).logProb(matrixX);
        assertArrayEquals(new double[]{-1, Double.NEGATIVE_INFINITY}, maskResult.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.5);
        ExponentialVertex tensorExponentialVertex = VertexOfType.exponential(0.5, 1.5);
        double expectedDensity = distribution.logDensity(2.0 - 0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorExponentialVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        ExponentialDistribution distribution = new ExponentialDistribution(1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(.75);
        ExponentialVertex ndExponentialVertex = VertexOfType.exponential(0.0, 1.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndExponentialVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        io.improbable.keanu.distributions.gradient.Exponential.Diff exponentialLogDiff = io.improbable.keanu.distributions.gradient.Exponential.dlnPdf(0.5, 2.5, 1.5);

        UniformVertex aTensor = VertexOfType.uniform(0.0, 5.0);
        aTensor.setValue(0.5);

        UniformVertex bTensor = VertexOfType.uniform(0.0, 5.0);
        bTensor.setValue(2.5);

        ExponentialVertex tensorExponentialVertex = VertexOfType.exponential(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorExponentialVertex.dLogProb(DoubleTensor.scalar(1.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(exponentialLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(exponentialLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(exponentialLogDiff.dPdx, actual.withRespectTo(tensorExponentialVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = VertexOfType.uniform(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ExponentialVertex vertexUnderTest = VertexOfType.exponential(
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
        ExponentialVertex exponential = VertexOfType.exponential(uniformA, ConstantVertex.of(3.0));

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.0),
            Nd4jDoubleTensor.scalar(0.9),
            0.1,
            uniformA,
            exponential,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = VertexOfType.uniform(1., 3.);
        ExponentialVertex exponential = VertexOfType.exponential(ConstantVertex.of(0.0), uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(2.5),
            0.1,
            uniformA,
            exponential,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void exponentialSampleMethodMatchesLogProbMethod() {

        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        ExponentialVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.LOCATION, 0.0)
            .withInput(ParameterName.LAMBDA, 0.5)
            .exponential();

        double from = 0.5;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueB = 2.0;

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
                .withInput(ParameterName.LOCATION, hyperParams.get(0))
                .withInput(ParameterName.LAMBDA, hyperParams.get(1))
                .exponential(),
            aB,
            latentAB,
            random
        );
    }

}
