package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.InverseGamma;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
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

public class TensorInverseGammaVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        TensorInverseGammaVertex tensorInverseGammaVertex = new TensorInverseGammaVertex(2.0, 1.0);
        double expectedDensity = InverseGamma.logPdf(2.0, 1.0, 0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorInverseGammaVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = InverseGamma.logPdf(2.0, 1.0, 0.25) + InverseGamma.logPdf(2.0, 1.0, 0.75);
        TensorInverseGammaVertex ndInverseGammaVertex = new TensorInverseGammaVertex(2.0, 1.0);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndInverseGammaVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        InverseGamma.Diff inverseGammaLogDiff = InverseGamma.dlnPdf(2.0, 1.0, 0.5);

        TensorUniformVertex aTensor = new TensorUniformVertex(0.0, 5.0);
        aTensor.setValue(2.0);

        TensorUniformVertex bTensor = new TensorUniformVertex(0.0, 1.0);
        bTensor.setValue(1.0);

        TensorInverseGammaVertex tensorInverseGammaVertex = new TensorInverseGammaVertex(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorInverseGammaVertex.dLogPdf(0.5);

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(inverseGammaLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(inverseGammaLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(inverseGammaLogDiff.dPdx, actual.withRespectTo(tensorInverseGammaVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 0.9, 0.3};

        TensorUniformVertex aTensor = new TensorUniformVertex(0.0, 1.0);
        aTensor.setValue(2.0);

        TensorUniformVertex bTensor = new TensorUniformVertex(0.0, 1.0);
        bTensor.setValue(1.0);

        Supplier<DoubleTensorVertex> vertexSupplier = () -> new TensorInverseGammaVertex(aTensor, bTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex a = new TensorUniformVertex(0.0, 1.0);
        a.setAndCascade(Nd4jDoubleTensor.scalar(2.5));
        TensorInverseGammaVertex vertexUnderTest = new TensorInverseGammaVertex(
            a,
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        TensorUniformVertex uniformA = new TensorUniformVertex(1.0, 3.0);
        TensorInverseGammaVertex inverseGamma = new TensorInverseGammaVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.9);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.5),
            Nd4jDoubleTensor.scalar(2.5),
            0.1,
            uniformA,
            inverseGamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        TensorUniformVertex uniformA = new TensorUniformVertex(1.0, 3.0);
        TensorInverseGammaVertex inverseGamma = new TensorInverseGammaVertex(3.0, uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.1);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.9);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.5),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            inverseGamma,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void inverseGammaSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        TensorInverseGammaVertex vertex = new TensorInverseGammaVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(2.0),
            new ConstantTensorVertex(3.0)
        );

        double from = 0.0;
        double to = 0.9;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 4.5;
        double trueB = 2.0;

        List<DoubleTensorVertex> aB = new ArrayList<>();
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueA)));
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueB)));

        List<DoubleTensorVertex> latentAB = new ArrayList<>();
        TensorUniformVertex latentA = new TensorUniformVertex(0.01, 10.0);
        latentA.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        TensorUniformVertex latentB = new TensorUniformVertex(0.01, 10.0);
        latentB.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentAB.add(latentA);
        latentAB.add(latentB);

        int numSamples = 2000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new TensorInverseGammaVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            aB,
            latentAB,
            random
        );
    }
}
