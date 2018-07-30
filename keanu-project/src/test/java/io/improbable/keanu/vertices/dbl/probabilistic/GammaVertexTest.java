package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Gamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class GammaVertexTest {

    private KeanuRandom random;

    private static final double DELTA = 0.0001;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        GammaDistribution distribution = new GammaDistribution(1.0, 1.5);
        GammaVertex tensorGamma = new GammaVertex(0.0, 1.5, 1.0);
        double expectedDensity = distribution.logDensity(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGamma, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        GammaDistribution distribution = new GammaDistribution(5.0, 1.0);
        double expectedLogDensity = distribution.logDensity(1) + distribution.logDensity(3);
        GammaVertex tensorGamma = new GammaVertex(0.0, 1., 5.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGamma, new double[]{1., 3.}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        Gamma.Diff gammaLogDiff = Gamma.dlnPdf(0.75, 2, 5.5, 1.5);

        UniformVertex aTensor = new UniformVertex(0.5, 1.0);
        aTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        UniformVertex thetaTensor = new UniformVertex(0.5, 1.0);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(2));

        UniformVertex kTensor = new UniformVertex(1.0, 5.0);
        kTensor.setValue(Nd4jDoubleTensor.scalar(5.5));

        GammaVertex tensorGamma = new GammaVertex(aTensor, thetaTensor, kTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorGamma.dLogPdf(Nd4jDoubleTensor.scalar(1.5));

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(gammaLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdtheta, actual.withRespectTo(thetaTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdk, actual.withRespectTo(kTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdx, actual.withRespectTo(tensorGamma.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{1.5, 2, 2.5, 3, 3.5};

        UniformVertex aTensor = new UniformVertex(0.5, 1.0);
        aTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        UniformVertex thetaTensor = new UniformVertex(0.5, 1.0);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        UniformVertex kTensor = new UniformVertex(1.0, 5.0);
        kTensor.setValue(Nd4jDoubleTensor.scalar(2.5));

        Supplier<DoubleVertex> vertexSupplier = () -> new GammaVertex(aTensor, thetaTensor, kTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex a = new UniformVertex(1.0, 2.0);
        a.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        GammaVertex vertexUnderTest = new GammaVertex(a, 1.5, 5.0);
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {

        UniformVertex uniformA = new UniformVertex(0.0, 1.0);
        GammaVertex gamma = new GammaVertex(uniformA, 2.0, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(3.);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.0),
            Nd4jDoubleTensor.scalar(2.0),
            0.1,
            uniformA,
            gamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdtheta() {

        UniformVertex uniformA = new UniformVertex(1.0, 3.0);
        GammaVertex gamma = new GammaVertex(0.0, uniformA, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(3.);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(2.5),
            0.1,
            uniformA,
            gamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdk() {

        UniformVertex uniformA = new UniformVertex(2.0, 5.0);
        GammaVertex gamma = new GammaVertex(0.0, 2.0, uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(3.);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(2.0),
            Nd4jDoubleTensor.scalar(4.5),
            0.1,
            uniformA,
            gamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void gammaSampledMethodMatchesLogProbMethod() {
        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        GammaVertex vertex = new GammaVertex(
            new int[]{sampleCount, 1},
            ConstantVertex.of(1.5),
            ConstantVertex.of(2.0),
            ConstantVertex.of(7.5)
        );

        double from = 1.5;
        double to = 2.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueTheta = 2.0;
        double trueK = 3.0;

        DoubleVertex constA = ConstantVertex.of(trueA);
        DoubleVertex constA2 = ConstantVertex.of(trueA);
        DoubleVertex constTheta = ConstantVertex.of(trueTheta);
        DoubleVertex constK = ConstantVertex.of(trueK);

        List<DoubleVertex> aThetaK = new ArrayList<>();
        aThetaK.add(constA);
        aThetaK.add(constTheta);
        aThetaK.add(constK);

        List<DoubleVertex> latentAThetaK = new ArrayList<>();
        UniformVertex latentTheta = new UniformVertex(0.01, 10.0);
        latentTheta.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        UniformVertex latentK = new UniformVertex(0.01, 10.0);
        latentK.setAndCascade(Nd4jDoubleTensor.scalar(0.1));

        latentAThetaK.add(constA2);
        latentAThetaK.add(latentTheta);
        latentAThetaK.add(latentK);

        int numSamples = 5000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new GammaVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), hyperParams.get(2)),
            aThetaK,
            latentAThetaK,
            random
        );

    }

}
