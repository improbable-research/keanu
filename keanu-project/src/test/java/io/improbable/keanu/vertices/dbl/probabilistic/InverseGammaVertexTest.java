package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.InverseGamma;
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

public class InverseGammaVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        InverseGammaVertex tensorInverseGammaVertex = new InverseGammaVertex(2.0, 1.0);
        double expectedDensity = InverseGamma.logPdf(2.0, 1.0, 0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorInverseGammaVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        double expectedLogDensity = InverseGamma.logPdf(2.0, 1.0, 0.25) + InverseGamma.logPdf(2.0, 1.0, 0.75);
        InverseGammaVertex ndInverseGammaVertex = new InverseGammaVertex(2.0, 1.0);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndInverseGammaVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        InverseGamma.Diff inverseGammaLogDiff = InverseGamma.dlnPdf(2.0, 1.0, 0.5);

        UniformVertex aTensor = new UniformVertex(0.0, 5.0);
        aTensor.setValue(2.0);

        UniformVertex bTensor = new UniformVertex(0.0, 1.0);
        bTensor.setValue(1.0);

        InverseGammaVertex tensorInverseGammaVertex = new InverseGammaVertex(aTensor, bTensor);
        Map<Long, DoubleTensor> actualDerivatives = tensorInverseGammaVertex.dLogPdf(0.5);

        PartialDerivatives actual = new PartialDerivatives(actualDerivatives);

        assertEquals(inverseGammaLogDiff.dPda, actual.withRespectTo(aTensor.getId()).scalar(), 1e-5);
        assertEquals(inverseGammaLogDiff.dPdb, actual.withRespectTo(bTensor.getId()).scalar(), 1e-5);
        assertEquals(inverseGammaLogDiff.dPdx, actual.withRespectTo(tensorInverseGammaVertex.getId()).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 0.9, 0.3};

        UniformVertex aTensor = new UniformVertex(0.0, 1.0);
        aTensor.setValue(2.0);

        UniformVertex bTensor = new UniformVertex(0.0, 1.0);
        bTensor.setValue(1.0);

        Supplier<DoubleVertex> vertexSupplier = () -> new InverseGammaVertex(aTensor, bTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setAndCascade(Nd4jDoubleTensor.scalar(2.5));
        InverseGammaVertex vertexUnderTest = new InverseGammaVertex(
            a,
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(1.0, 3.0);
        InverseGammaVertex inverseGamma = new InverseGammaVertex(uniformA, 3.0);

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
        UniformVertex uniformA = new UniformVertex(1.0, 3.0);
        InverseGammaVertex inverseGamma = new InverseGammaVertex(3.0, uniformA);

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
        InverseGammaVertex vertex = new InverseGammaVertex(
            new int[]{sampleCount, 1},
            2.0,
            3.0
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

        List<DoubleVertex> aB = new ArrayList<>();
        aB.add(ConstantVertex.of(trueA));
        aB.add(ConstantVertex.of(trueB));

        List<DoubleVertex> latentAB = new ArrayList<>();
        UniformVertex latentA = new UniformVertex(0.01, 10.0);
        latentA.setAndCascade(Nd4jDoubleTensor.scalar(9.9));
        UniformVertex latentB = new UniformVertex(0.01, 10.0);
        latentB.setAndCascade(Nd4jDoubleTensor.scalar(0.1));
        latentAB.add(latentA);
        latentAB.add(latentB);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new InverseGammaVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            aB,
            latentAB,
            random
        );
    }
}
