package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.distributions.continuous.Logistic;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.*;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorPartialDerivatives;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.vertices.dbltensor.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TensorLogisticVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        TensorLogisticVertex tensorLogisticVertex = new TensorLogisticVertex(0.5, 1.5);
        double expectedDensity = Logistic.logPdf(0.5, 1.5, 2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorLogisticVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = Logistic.logPdf(0.0, 1.0, 0.25) + Logistic.logPdf(0.0, 1.0, .75);
        TensorLogisticVertex ndLogisticVertex = new TensorLogisticVertex(0.0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndLogisticVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Logistic.Diff logisticLogDiff = Logistic.dlnPdf(0.0, 0.5, 1.5);

        TensorUniformVertex aTensor = new TensorUniformVertex(0.0, 5.0);
        aTensor.setValue(0.0);

        TensorUniformVertex bTensor = new TensorUniformVertex(0.0, 5.0);
        bTensor.setValue(0.5);

        TensorLogisticVertex tensorLogisticVertex = new TensorLogisticVertex(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorLogisticVertex.dLogPdf(1.5);

        TensorPartialDerivatives actual = new TensorPartialDerivatives(actualDerivatives);

        assertEquals(logisticLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(logisticLogDiff.dPdx, actual.withRespectTo(tensorLogisticVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        TensorUniformVertex mu = new TensorUniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        TensorLogisticVertex vertexUnderTest = new TensorLogisticVertex(
            mu,
            3.
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        TensorUniformVertex uniformA = new TensorUniformVertex(0.0, 1.0);
        TensorLogisticVertex logistic = new TensorLogisticVertex(uniformA, 1.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.0),
            Nd4jDoubleTensor.scalar(0.9),
            0.1,
            uniformA,
            logistic,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        TensorUniformVertex uniformA = new TensorUniformVertex(0., 1.);
        TensorLogisticVertex logistic = new TensorLogisticVertex(0.0, uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(1.0);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.5),
            Nd4jDoubleTensor.scalar(3.5),
            0.1,
            uniformA,
            logistic,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void logisticSampleMethodMatchesLogProbMethod() {

        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        TensorLogisticVertex vertex = new TensorLogisticVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(0.0),
            new ConstantTensorVertex(0.5)
        );

        double from = 0.5;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 0.0;
        double trueB = 0.5;

        List<DoubleTensorVertex> aB = new ArrayList<>();
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueA)));
        aB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueB)));

        List<DoubleTensorVertex> latentAB = new ArrayList<>();
        TensorUniformVertex latentB = new TensorUniformVertex(0.01, 10.0);
        latentB.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentAB.add(new ConstantTensorVertex(Nd4jDoubleTensor.scalar(trueA)));
        latentAB.add(latentB);

        int numSamples = 2000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new TensorLogisticVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            aB,
            latentAB,
            random
        );
    }

}
