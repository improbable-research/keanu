package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.continuous.Exponential;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ExponentialVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void maskAppliedToValuesWhereXLessThanA() {
        DoubleTensor matrixX = Nd4jDoubleTensor.create(new double[]{1, -2}, new int[]{2, 1});

        DoubleTensor maskResult = Exponential.withParameters(DoubleTensor.ONE_SCALAR).logProb(matrixX);
        assertArrayEquals(new double[]{-1, Double.NEGATIVE_INFINITY}, maskResult.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        ExponentialDistribution distribution = new ExponentialDistribution(1.5);
        ExponentialVertex tensorExponentialVertex = new ExponentialVertex(1.5);
        double expectedDensity = distribution.logDensity(2.0);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorExponentialVertex, 2.0, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        ExponentialDistribution distribution = new ExponentialDistribution(1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(.75);
        ExponentialVertex ndExponentialVertex = new ExponentialVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(ndExponentialVertex, new double[]{0.25, .75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        io.improbable.keanu.distributions.gradient.Exponential.Diff exponentialLogDiff = io.improbable.keanu.distributions.gradient.Exponential.dlnPdf(2.5, 1.5);

        UniformVertex bTensor = new UniformVertex(0.0, 5.0);
        bTensor.setValue(2.5);

        ExponentialVertex tensorExponentialVertex = new ExponentialVertex(bTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorExponentialVertex.dLogPdf(1.5, bTensor, tensorExponentialVertex);

        assertEquals(exponentialLogDiff.dPdlambda, actualDerivatives.get(bTensor).scalar(), 1e-5);
        assertEquals(exponentialLogDiff.dPdx, actualDerivatives.get(tensorExponentialVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {
        double[] vector = new double[]{1.0, 1.25, 1.5, 3.0, 5.0};

        UniformVertex lambdaVertex = new UniformVertex(0.0, 1.0);
        lambdaVertex.setValue(1.0);

        Supplier<ExponentialVertex> vertexSupplier = () -> new ExponentialVertex(lambdaVertex);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex mu = new UniformVertex(0.0, 1.0);
        mu.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        ExponentialVertex vertexUnderTest = new ExponentialVertex(
            3.
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformA = new UniformVertex(1., 3.);
        ExponentialVertex exponential = new ExponentialVertex(uniformA);

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
        ExponentialVertex vertex = new ExponentialVertex(
            new int[]{sampleCount, 1},
            ConstantVertex.of(0.5)
        );

        double from = 0.5;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueLambda = 2.0;

        List<DoubleVertex> generationParameters = new ArrayList<>();
        generationParameters.add(ConstantVertex.of(trueLambda));

        List<DoubleVertex> latents = new ArrayList<>();
        UniformVertex latentLambda = new UniformVertex(0.01, 10.0);
        latentLambda.setAndCascade(0.1);
        latents.add(latentLambda);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new ExponentialVertex(new int[]{numSamples, 1}, hyperParams.get(0)),
            generationParameters,
            latents,
            random
        );
    }

}
