package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.Exponential;
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

public class TensorExponentialVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        TensorExponentialVertex tensorExponentialVertex = new TensorExponentialVertex(1, 1.5, new KeanuRandom(1));
        double expectedDensity = Exponential.logPdf(1, 1.5, 2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorExponentialVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = Exponential.logPdf(0.0, 1.0, 0.25) + Exponential.logPdf(0.0, 1.0, .75);
        TensorExponentialVertex ndExponentialVertex = new TensorExponentialVertex(0, 1, new KeanuRandom(1));
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndExponentialVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Exponential.Diff exponentialLogDiff = Exponential.dlnPdf(2, 2.5, 1.5);

        KeanuRandom keanuRandom = new KeanuRandom(1);

        TensorUniformVertex aTensor = new TensorUniformVertex(0.0, 5.0, keanuRandom);
        aTensor.setValue(2.);

        TensorUniformVertex bTensor = new TensorUniformVertex(0.0, 5.0, keanuRandom);
        bTensor.setValue(2.5);

        TensorExponentialVertex tensorExponentialVertex = new TensorExponentialVertex(aTensor, bTensor, keanuRandom);
        Map<Long, DoubleTensor> actualDerivatives = tensorExponentialVertex.dLogPdf(1.5);

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(exponentialLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(exponentialLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(exponentialLogDiff.dPdx, actual.withRespectTo(tensorExponentialVertex.getId()).scalar(), 1e-5);
    }

//    @Test
//    public void matchesKnownDerivativeLogDensityOfVector() {
//
//        double[] vector = new double[]{1, 1.5, 2, 2.5};
//
//        KeanuRandom keanuRandom = new KeanuRandom(1);
//
//        TensorUniformVertex aTensor = new TensorUniformVertex(0.0, 10., keanuRandom);
//        aTensor.setValue(5.);
//
//        TensorUniformVertex bTensor = new TensorUniformVertex(0.0, 10., keanuRandom);
//        bTensor.setValue(5.);
//
//        Supplier<DoubleTensorVertex> vertexSupplier = () -> new TensorExponentialVertex(aTensor, bTensor, keanuRandom);
//
//        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
//    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex mu = new TensorUniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        TensorExponentialVertex vertexUnderTest = new TensorExponentialVertex(
            mu,
            new ConstantTensorVertex(3.0),
            random
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        TensorUniformVertex uniformA = new TensorUniformVertex(new ConstantTensorVertex(0.0), new ConstantTensorVertex(1.0), random);
        TensorExponentialVertex exponential = new TensorExponentialVertex(uniformA, new ConstantTensorVertex(3.0), random);

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
        TensorUniformVertex uniformA = new TensorUniformVertex(new ConstantTensorVertex(1.), new ConstantTensorVertex(3.), random);
        TensorExponentialVertex exponential = new TensorExponentialVertex(new ConstantTensorVertex(0.0), uniformA, random);

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
        TensorExponentialVertex vertex = new TensorExponentialVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(0.0),
            new ConstantTensorVertex(0.5),
            random
        );

        double from = 0.5;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueB = 2.0;

        List<DoubleTensorVertex> aB = new ArrayList<>();
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueA)));
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueB)));

        List<DoubleTensorVertex> latentAB = new ArrayList<>();
        TensorUniformVertex latentB = new TensorUniformVertex(0.01, 10.0, random);
        latentB.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentAB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueA)));
        latentAB.add(latentB);

        int numSamples = 2000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new TensorExponentialVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), random),
            aB,
            latentAB
        );
    }

}
