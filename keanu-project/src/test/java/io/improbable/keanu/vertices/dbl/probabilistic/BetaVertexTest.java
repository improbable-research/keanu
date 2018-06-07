package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Beta;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class BetaVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        BetaVertex tensorBetaVertex = new BetaVertex(2., 3.);
        double expectedDensity = Beta.logPdf(2.0, 3.0, 0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorBetaVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = Beta.logPdf(2.0, 3.0, 0.25) + Beta.logPdf(2.0, 3.0, 0.1);
        BetaVertex ndBetaVertex = new BetaVertex(2, 3);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndBetaVertex, new double[]{0.25, 0.1}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Beta.Diff betaLogDiff = Beta.dlnPdf(2.0, 3.0, 0.5);

        UniformVertex alphaTensor = new UniformVertex(0.0, 5.0);
        alphaTensor.setValue(2.0);

        UniformVertex betaTensor = new UniformVertex(0.0, 5.0);
        betaTensor.setValue(3.0);

        BetaVertex tensorBetaVertex = new BetaVertex(alphaTensor, betaTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorBetaVertex.dLogPdf(0.5);

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(betaLogDiff.dPdalpha, actual.withRespectTo(alphaTensor.getId()).scalar(), 1e-5);
        assertEquals(betaLogDiff.dPdbeta, actual.withRespectTo(betaTensor.getId()).scalar(), 1e-5);
        assertEquals(betaLogDiff.dPdx, actual.withRespectTo(tensorBetaVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 0.9, 0.3};

        UniformVertex alphaTensor = new UniformVertex(0.0, 5.0);
        alphaTensor.setValue(2.0);

        UniformVertex betaTensor = new UniformVertex(0.0, 5.0);
        betaTensor.setValue(3.0);

        Supplier<DoubleVertex> vertexSupplier = () -> new BetaVertex(alphaTensor, betaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex alpha = new UniformVertex(0.0, 1.0);
        alpha.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        BetaVertex vertexUnderTest = new BetaVertex(
            alpha,
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdalpha() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        BetaVertex beta = new BetaVertex(uniformA, 3.0);

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
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        BetaVertex beta = new BetaVertex(3.0, uniformA);

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
        BetaVertex vertex = new BetaVertex(
            new int[]{sampleCount, 1},
            5.0,
            2.0
        );

        double from = 0.3;
        double to = 0.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void betaSampleMethodMatchesLogProbMethodForAlphaLessthanBeta() {

        int sampleCount = 1000000;
        BetaVertex vertex = new BetaVertex(
            new int[]{sampleCount, 1},
            5.0,
            2.0
        );

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
        UniformVertex latentAlpha = new UniformVertex(0.01, 10.0);
        latentAlpha.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentBeta = new UniformVertex(0.01, 10.0);
        latentBeta.setAndCascade(DoubleTensor.scalar(0.1));
        latentAlphaBeta.add(latentAlpha);
        latentAlphaBeta.add(latentBeta);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new BetaVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            alphaBeta,
            latentAlphaBeta,
            random
        );
    }
}
