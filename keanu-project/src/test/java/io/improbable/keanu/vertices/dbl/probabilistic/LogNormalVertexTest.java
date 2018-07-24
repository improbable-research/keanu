package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.distributions.gradient.LogNormal;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class LogNormalVertexTest {
    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        LogNormalDistribution distribution = new LogNormalDistribution(0, 1);
        LogNormalVertex tensorLogNormalVertex = VertexOfType.logNormal(0., 1.);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogNormalVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        LogNormalDistribution distribution = new LogNormalDistribution(0, 1);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75);
        LogNormalVertex tensorLogNormalVertex = VertexOfType.logNormal(0., 1.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorLogNormalVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        LogNormal.Diff logNormalLogDiff = LogNormal.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = VertexOfType.uniform(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        LogNormalVertex tensorLogNormalVertex = VertexOfType.logNormal(muTensor, sigmaTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorLogNormalVertex.dLogProb(DoubleTensor.scalar(0.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(logNormalLogDiff.dPdmu, actual.withRespectTo(muTensor.getId()).scalar(), 1e-5);
        assertEquals(logNormalLogDiff.dPdsigma, actual.withRespectTo(sigmaTensor.getId()).scalar(), 1e-5);
        assertEquals(logNormalLogDiff.dPdx, actual.withRespectTo(tensorLogNormalVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, -0.75, 0.1, -2, 1.3};

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = VertexOfType.uniform(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, () -> VertexOfType.logNormal(muTensor, sigmaTensor));
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = VertexOfType.uniform(0.0, 1.0);
        mu.setAndCascade(0.5);
        LogNormalVertex vertexUnderTest = VertexOfType.logNormal(mu, ConstantVertex.of(3.0));
        vertexUnderTest.setAndCascade(1.0);
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        LogNormalVertex logNormal = VertexOfType.logNormal(uniformA, ConstantVertex.of(3.0));

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(1.5),
            0.1,
            uniformA,
            logNormal,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        LogNormalVertex logNormal = VertexOfType.logNormal(ConstantVertex.of(3.0), uniformA);

        DoubleTensor vertexStartValue = DoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = DoubleTensor.scalar(1.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            logNormal,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void logNormalSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        LogNormalVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.MU, 0.0)
            .withInput(ParameterName.SIGMA, 2.0)
            .logNormal();

        double from = 0.1;
        double to = 8;
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
        UniformVertex latentMu = VertexOfType.uniform(0.01, 10.0);
        latentMu.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentSigma = VertexOfType.uniform(0.01, 10.0);
        latentSigma.setAndCascade(DoubleTensor.scalar(0.1));
        latentMuSigma.add(latentMu);
        latentMuSigma.add(latentSigma);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new DistributionVertexBuilder()
                .shaped(numSamples, 1)
                .withInput(ParameterName.MU, hyperParams.get(0))
                .withInput(ParameterName.SIGMA, hyperParams.get(1))
                .logNormal(),
            muSigma,
            latentMuSigma,
            random
        );
    }
}
