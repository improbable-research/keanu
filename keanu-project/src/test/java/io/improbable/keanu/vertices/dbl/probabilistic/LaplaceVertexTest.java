package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.LaplaceDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.distributions.gradient.Laplace;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        LaplaceVertex tensorLaplaceVertex = VertexOfType.laplace(0.5, 1.);
        LaplaceDistribution distribution = new LaplaceDistribution(0.5, 1.0);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLaplaceVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        LaplaceDistribution distribution = new LaplaceDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LaplaceVertex ndLaplaceVertex = VertexOfType.laplace(0., 1.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLaplaceVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Laplace.Diff laplaceLogDiff = Laplace.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex betaTensor = VertexOfType.uniform(0.0, 1.0);
        betaTensor.setValue(1.0);

        LaplaceVertex tensorLaplaceVertex = VertexOfType.laplace(muTensor, betaTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorLaplaceVertex.dLogProb(DoubleTensor.scalar(0.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(laplaceLogDiff.dPdmu, actual.withRespectTo(muTensor.getId()).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdbeta, actual.withRespectTo(betaTensor.getId()).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdx, actual.withRespectTo(tensorLaplaceVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 22, 1.3};

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex betaTensor = VertexOfType.uniform(0.0, 1.0);
        betaTensor.setValue(1.0);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, () -> VertexOfType.laplace(muTensor, betaTensor));
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = VertexOfType.uniform(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        LaplaceVertex vertexUnderTest = VertexOfType.laplace(
            mu,
            ConstantVertex.of(3.0)
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        LaplaceVertex laplace = VertexOfType.laplace(uniformA, ConstantVertex.of(3.0));

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(1.5),
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
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        LaplaceVertex laplace = VertexOfType.laplace(ConstantVertex.of(3.0), uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            laplace,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void laplaceSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        LaplaceVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.MU, 0.0)
            .withInput(ParameterName.BETA, 2.0)
            .laplace();

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
        UniformVertex latentMu = VertexOfType.uniform(0.01, 10.0);
        latentMu.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        UniformVertex latentBeta = VertexOfType.uniform(0.01, 10.0);
        latentBeta.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentMuBeta.add(latentMu);
        latentMuBeta.add(latentBeta);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new DistributionVertexBuilder()
                .shaped(numSamples, 1)
                .withInput(ParameterName.MU, hyperParams.get(0))
                .withInput(ParameterName.BETA, hyperParams.get(1))
                .laplace(),
            muBeta,
            latentMuBeta,
            random
        );
    }
}
