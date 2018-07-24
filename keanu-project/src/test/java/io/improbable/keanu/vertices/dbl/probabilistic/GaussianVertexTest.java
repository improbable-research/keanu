package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.distributions.gradient.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class GaussianVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        GaussianVertex tensorGaussianVertex = VertexOfType.gaussian(0., 1.);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(-0.75);
        GaussianVertex tensorGaussianVertex = VertexOfType.gaussian(0., 1.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{0.25, -0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Gaussian.Diff gaussianLogDiff = Gaussian.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = VertexOfType.uniform(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        GaussianVertex tensorGaussianVertex = VertexOfType.gaussian(muTensor, sigmaTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorGaussianVertex.dLogProb(DoubleTensor.scalar(0.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(gaussianLogDiff.dPdmu, actual.withRespectTo(muTensor.getId()).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdsigma, actual.withRespectTo(sigmaTensor.getId()).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdx, actual.withRespectTo(tensorGaussianVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, -0.75, 0.1, -2, 1.3};

        UniformVertex muTensor = VertexOfType.uniform(0.0, 1.0);
        muTensor.setValue(0.0);

        UniformVertex sigmaTensor = VertexOfType.uniform(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, () -> VertexOfType.gaussian(muTensor, sigmaTensor));
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = VertexOfType.uniform(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        GaussianVertex vertexUnderTest = VertexOfType.gaussian(
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
        GaussianVertex gaussian = VertexOfType.gaussian(uniformA, ConstantVertex.of(3.0));

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(1.5),
            0.1,
            uniformA,
            gaussian,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = VertexOfType.uniform(1.5, 3.0);
        GaussianVertex gaussian = VertexOfType.gaussian(ConstantVertex.of(3.0), uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            gaussian,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        GaussianVertex vertex = new DistributionVertexBuilder()
            .shaped(new int[]{sampleCount, 1})
            .withInput(ParameterName.MU, 0.0)
            .withInput(ParameterName.SIGMA, 2.0)
            .gaussian();

        double from = -4;
        double to = 4;
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
                .shaped(new int[]{numSamples, 1})
            .withInput(ParameterName.MU, hyperParams.get(0))
            .withInput(ParameterName.SIGMA, hyperParams.get(1))
            .gaussian(),
            muSigma,
            latentMuSigma,
            random
        );
    }
}
