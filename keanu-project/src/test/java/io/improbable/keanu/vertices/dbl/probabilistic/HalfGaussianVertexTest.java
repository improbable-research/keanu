package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertexFactory;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class HalfGaussianVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        double expectedDensity = distribution.logDensity(0.5) + Math.log(2);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfNegativeScalar() {

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGaussianVertex, -0.5, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        NormalDistribution distribution = new NormalDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75) + 2 * Math.log(2);
        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownLogDensityOfNegativeVector() {

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGaussianVertex, new double[]{-0.25, 0.75}, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Gaussian.Diff gaussianLogDiff = Gaussian.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        HalfGaussianVertex tensorGaussianVertex = new HalfGaussianVertex(sigmaTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorGaussianVertex.dLogPdf(0.5, sigmaTensor, tensorGaussianVertex);

        assertEquals(gaussianLogDiff.dPdsigma, actualDerivatives.get(sigmaTensor).scalar(), 1e-5);
        assertEquals(gaussianLogDiff.dPdx, actualDerivatives.get(tensorGaussianVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 2, 1.3};

        UniformVertex sigmaTensor = new UniformVertex(0.0, 1.0);
        sigmaTensor.setValue(1.0);

        Supplier<HalfGaussianVertex> vertexSupplier = () -> new HalfGaussianVertex(sigmaTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        HalfGaussianVertex vertexUnderTest = new HalfGaussianVertex(
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdsigma() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        HalfGaussianVertex gaussian = new HalfGaussianVertex(uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(0.0),
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
        HalfGaussianVertex vertex = new HalfGaussianVertex(
            new long[]{sampleCount, 1},
            ConstantVertexFactory.of(2.0)
        );

        double from = 0;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueSigma = 2.0;

        List<DoubleVertex> sigma = new ArrayList<>();
        sigma.add(ConstantVertexFactory.of(trueSigma));

        List<DoubleVertex> latentSigmaList = new ArrayList<>();
        UniformVertex latentSigma = new UniformVertex(0.01, 10.0);
        latentSigma.setAndCascade(DoubleTensor.scalar(0.1));
        latentSigmaList.add(latentSigma);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new HalfGaussianVertex(new long[]{numSamples, 1}, hyperParams.get(0)),
            sigma,
            latentSigmaList,
            random
        );
    }
}
