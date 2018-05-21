package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.Gamma;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbltensor.*;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class TensorGammaVertexTest {

    private KeanuRandom random;

    private static final double DELTA = 0.0001;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        GammaVertex gamma = new GammaVertex(0.5, 1, 1.5);
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.5, 1, 1.5);

        double expectedDensity = gamma.logPdf(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGamma, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        double expectedLogDensity = Gamma.logPdf(0.0, 1.0, 5., 1) + Gamma.logPdf(0.0, 1.0, 5., 3);
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.0, 1., 5.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGamma, new double[]{1., 3.}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        KeanuRandom keanuRandom = new KeanuRandom(1);

        Gamma.Diff gammaLogDiff = Gamma.dlnPdf(0.75, 2, 5.5, 1.5);

        TensorUniformVertex aTensor = new TensorUniformVertex(0.5, 1.0);
        aTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        TensorUniformVertex thetaTensor = new TensorUniformVertex(0.5, 1.0);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(2));

        TensorUniformVertex kTensor = new TensorUniformVertex(1.0, 5.0);
        kTensor.setValue(Nd4jDoubleTensor.scalar(5.5));

        TensorGammaVertex tensorGamma = new TensorGammaVertex(aTensor, thetaTensor, kTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorGamma.dLogPdf(Nd4jDoubleTensor.scalar(1.5));

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(gammaLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdtheta, actual.withRespectTo(thetaTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdk, actual.withRespectTo(kTensor.getId()).scalar(), 1e-5);
        assertEquals(gammaLogDiff.dPdx, actual.withRespectTo(tensorGamma.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{1.5, 2, 2.5, 3, 3.5};

        KeanuRandom keanuRandom = new KeanuRandom(1);

        TensorUniformVertex aTensor = new TensorUniformVertex(0.5, 1.0);
        aTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        TensorUniformVertex thetaTensor = new TensorUniformVertex(0.5, 1.0);
        thetaTensor.setValue(Nd4jDoubleTensor.scalar(0.75));

        TensorUniformVertex kTensor = new TensorUniformVertex(1.0, 5.0);
        kTensor.setValue(Nd4jDoubleTensor.scalar(2.5));

        Supplier<DoubleTensorVertex> vertexSupplier = () -> new TensorGammaVertex(aTensor, thetaTensor, kTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex a = new TensorUniformVertex(1.0, 2.0);
        a.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        TensorGammaVertex vertexUnderTest = new TensorGammaVertex(
            a,
            new ConstantTensorVertex(1.5),
            new ConstantTensorVertex(5.0)
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {

        TensorUniformVertex uniformA = new TensorUniformVertex(0.0, 1.0);
        TensorGammaVertex gamma = new TensorGammaVertex(uniformA, 2.0, 3.0);

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

        TensorUniformVertex uniformA = new TensorUniformVertex(1.0, 3.0);
        TensorGammaVertex gamma = new TensorGammaVertex(0.0, uniformA, 3.0);

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

        TensorUniformVertex uniformA = new TensorUniformVertex(2.0, 5.0);
        TensorGammaVertex gamma = new TensorGammaVertex(0.0, 2.0, uniformA);

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
        TensorGammaVertex vertex = new TensorGammaVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(1.5),
            new ConstantTensorVertex(2.0),
            new ConstantTensorVertex(7.5)
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

        DoubleTensorVertex constA = new ConstantTensorVertex(trueA);
        DoubleTensorVertex constA2 = new ConstantTensorVertex(trueA);
        DoubleTensorVertex constTheta = new ConstantTensorVertex(trueTheta);
        DoubleTensorVertex constK = new ConstantTensorVertex(trueK);

        List<DoubleTensorVertex> aThetaK = new ArrayList<>();
        aThetaK.add(constA);
        aThetaK.add(constTheta);
        aThetaK.add(constK);

        List<DoubleTensorVertex> latentAThetaK = new ArrayList<>();
        TensorUniformVertex latentTheta = new TensorUniformVertex(0.01, 10.0);
        latentTheta.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        TensorUniformVertex latentK = new TensorUniformVertex(0.01, 10.0);
        latentK.setAndCascade(Nd4jDoubleTensor.scalar(0.1));

        latentAThetaK.add(constA2);
        latentAThetaK.add(latentTheta);
        latentAThetaK.add(latentK);

        int numSamples = 5000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new TensorGammaVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), hyperParams.get(2)),
            aThetaK,
            latentAThetaK,
            random
        );

    }

}
