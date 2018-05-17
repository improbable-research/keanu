package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.Laplace;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
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

public class TensorLaplaceVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        TensorLaplaceVertex tensorLaplaceVertex = new TensorLaplaceVertex(0.5, 1, new KeanuRandom(1));
        double expectedDensity = Laplace.logPdf(0.5, 1.0, 0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLaplaceVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = Laplace.logPdf(0.0, 1.0, 0.25) + Laplace.logPdf(0.0, 1.0, 0.75);
        TensorLaplaceVertex ndLaplaceVertex = new TensorLaplaceVertex(0, 1, new KeanuRandom(1));
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLaplaceVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Laplace.Diff laplaceLogDiff = Laplace.dlnPdf(0.0, 1.0, 0.5);

        KeanuRandom keanuRandom = new KeanuRandom(1);

        TensorUniformVertex muTensor = new TensorUniformVertex(0.0, 1.0, keanuRandom);
        muTensor.setValue(0.0);

        TensorUniformVertex betaTensor = new TensorUniformVertex(0.0, 1.0, keanuRandom);
        betaTensor.setValue(1.0);

        TensorLaplaceVertex tensorLaplaceVertex = new TensorLaplaceVertex(muTensor, betaTensor, keanuRandom);
        Map<Long, DoubleTensor> actualDerivatives = tensorLaplaceVertex.dLogPdf(0.5);

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(laplaceLogDiff.dPdmu, actual.withRespectTo(muTensor.getId()).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdbeta, actual.withRespectTo(betaTensor.getId()).scalar(), 1e-5);
        assertEquals(laplaceLogDiff.dPdx, actual.withRespectTo(tensorLaplaceVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 22, 1.3};

        KeanuRandom keanuRandom = new KeanuRandom(1);

        TensorUniformVertex muTensor = new TensorUniformVertex(0.0, 1.0, keanuRandom);
        muTensor.setValue(0.0);

        TensorUniformVertex betaTensor = new TensorUniformVertex(0.0, 1.0, keanuRandom);
        betaTensor.setValue(1.0);

        Supplier<DoubleTensorVertex> vertexSupplier = () -> new TensorLaplaceVertex(muTensor, betaTensor, keanuRandom);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex mu = new TensorUniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        TensorLaplaceVertex vertexUnderTest = new TensorLaplaceVertex(
            mu,
            new ConstantTensorVertex(3.0),
            random
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdmu() {
        TensorUniformVertex uniformA = new TensorUniformVertex(new ConstantTensorVertex(1.5), new ConstantTensorVertex(3.0), random);
        TensorLaplaceVertex laplace = new TensorLaplaceVertex(uniformA, new ConstantTensorVertex(3.0), random);

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
        TensorUniformVertex uniformA = new TensorUniformVertex(new ConstantTensorVertex(1.5), new ConstantTensorVertex(3.0), random);
        TensorLaplaceVertex laplace = new TensorLaplaceVertex(new ConstantTensorVertex(3.0), uniformA, random);

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

        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        TensorLaplaceVertex vertex = new TensorLaplaceVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(0.0),
            new ConstantTensorVertex(2.0),
            random
        );

        double from = -4;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 4.5;
        double trueBeta = 2.0;

        List<DoubleTensorVertex> muBeta = new ArrayList<>();
        muBeta.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueMu)));
        muBeta.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueBeta)));

        List<DoubleTensorVertex> latentMuBeta = new ArrayList<>();
        TensorUniformVertex latentMu = new TensorUniformVertex(0.01, 10.0, random);
        latentMu.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        TensorUniformVertex latentBeta = new TensorUniformVertex(0.01, 10.0, random);
        latentBeta.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentMuBeta.add(latentMu);
        latentMuBeta.add(latentBeta);

        int numSamples = 2000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new TensorLaplaceVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), random),
            muBeta,
            latentMuBeta
        );
    }
}
