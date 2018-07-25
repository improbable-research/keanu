package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.distributions.gradient.Beta;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class BetaVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        BetaVertex tensorBetaVertex = VertexOfType.beta(2., 3.);
        BetaDistribution betaDistribution = new BetaDistribution(2.0, 3.0);
        double expectedDensity = betaDistribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorBetaVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        BetaDistribution betaDistribution = new BetaDistribution(2, 3);
        double expectedLogDensity = betaDistribution.logDensity(0.25) + betaDistribution.logDensity(0.1);
        BetaVertex ndBetaVertex = VertexOfType.beta(2., 3.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndBetaVertex, new double[]{0.25, 0.1}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Beta.Diff betaLogDiff = Beta.dlnPdf(2.0, 3.0, 0.5);

        UniformVertex alphaTensor = VertexOfType.uniform(0.0, 5.0);
        alphaTensor.setValue(2.0);

        UniformVertex betaTensor = VertexOfType.uniform(0.0, 5.0);
        betaTensor.setValue(3.0);

        BetaVertex tensorBetaVertex = VertexOfType.beta(alphaTensor, betaTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorBetaVertex.dLogProb(DoubleTensor.scalar(0.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(betaLogDiff.dPdalpha, actual.withRespectTo(alphaTensor.getId()).scalar(), 1e-5);
        assertEquals(betaLogDiff.dPdbeta, actual.withRespectTo(betaTensor.getId()).scalar(), 1e-5);
        assertEquals(betaLogDiff.dPdx, actual.withRespectTo(tensorBetaVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 0.9, 0.3};

        UniformVertex alphaTensor = VertexOfType.uniform(0.0, 5.0);
        alphaTensor.setValue(2.0);

        UniformVertex betaTensor = VertexOfType.uniform(0.0, 5.0);
        betaTensor.setValue(3.0);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, () -> VertexOfType.beta(alphaTensor, betaTensor));
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex alpha = VertexOfType.uniform(0.0, 1.0);
        alpha.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        BetaVertex vertexUnderTest = VertexOfType.beta(
            alpha,
            ConstantVertex.of(3.0)
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdalpha() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        BetaVertex beta = VertexOfType.beta(uniformA, ConstantVertex.of(3.0));

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.9);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.6),
            Nd4jDoubleTensor.scalar(2.9),
            0.1,
            uniformA,
            beta,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdbeta() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        BetaVertex beta = VertexOfType.beta(ConstantVertex.of(3.0), uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            beta,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void betaSampleMethodMatchesLogProbMethodForAlphaGreaterThanBeta() {

        int sampleCount = 1000000;
        BetaVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.A, 5.0)
            .withInput(ParameterName.B, 2.0)
            .beta();

        double from = 0.3;
        double to = 0.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void betaSampleMethodMatchesLogProbMethodForAlphaLessThanBeta() {

        int sampleCount = 1100000;
        BetaVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.A, 2.0)
            .withInput(ParameterName.B, 5.0)
            .beta();

        double from = 0.3;
        double to = 0.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueAlpha = 2.;
        double trueBeta = 2.;

        List<DoubleVertex> alphaBeta = new ArrayList<>();
        alphaBeta.add(ConstantVertex.of(trueAlpha));
        alphaBeta.add(ConstantVertex.of(trueBeta));

        List<DoubleVertex> latentAlphaBeta = new ArrayList<>();
        UniformVertex latentAlpha = VertexOfType.uniform(0.01, 10.0);
        latentAlpha.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentBeta = VertexOfType.uniform(0.01, 10.0);
        latentBeta.setAndCascade(DoubleTensor.scalar(0.1));
        latentAlphaBeta.add(latentAlpha);
        latentAlphaBeta.add(latentBeta);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new DistributionVertexBuilder().shaped(numSamples, 1)
            .withInput(ParameterName.A, hyperParams.get(0))
            .withInput(ParameterName.B, hyperParams.get(1))
            .beta(),
            alphaBeta,
            latentAlphaBeta,
            random
        );
    }
}
