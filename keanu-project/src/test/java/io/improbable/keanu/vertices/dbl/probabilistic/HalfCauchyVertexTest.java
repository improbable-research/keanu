package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Cauchy;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.CauchyDistribution;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class HalfCauchyVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        double expectedDensity = distribution.logDensity(0.5) + Math.log(2.0);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorHalfCauchyVertex, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfNegativeScalar() {

        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorHalfCauchyVertex, -0.5, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(0.75) + 2.0 * Math.log(2.0);
        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorHalfCauchyVertex, new double[]{0.25, 0.75}, expectedLogDensity);
    }

    @Test
    public void matchesKnownLogDensityOfNegativeVector() {

        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorHalfCauchyVertex, new double[]{-0.25, 0.75}, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Cauchy.Diff cauchyLogDiff = Cauchy.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        HalfCauchyVertex tensorHalfCauchyVertex = new HalfCauchyVertex(scaleTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorHalfCauchyVertex.dLogPdf(0.5, scaleTensor, tensorHalfCauchyVertex);

        assertEquals(cauchyLogDiff.dPdscale, actualDerivatives.get(scaleTensor).scalar(), 1e-5);
        assertEquals(cauchyLogDiff.dPdx, actualDerivatives.get(tensorHalfCauchyVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, 0.75, 0.1, 2, 1.3};

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        Supplier<HalfCauchyVertex> vertexSupplier = () -> new HalfCauchyVertex(scaleTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        HalfCauchyVertex vertexUnderTest = new HalfCauchyVertex(3.0);
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdscale() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        HalfCauchyVertex cauchy = new HalfCauchyVertex(uniformA);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(0.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(1.0),
            Nd4jDoubleTensor.scalar(3.0),
            0.1,
            uniformA,
            cauchy,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Category(Slow.class)
    @Test
    public void cauchySampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        HalfCauchyVertex vertex = new HalfCauchyVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(2.0)
        );

        double from = 0;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueScale = 2.0;

        List<DoubleVertex> scale = new ArrayList<>();
        scale.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentScaleList = new ArrayList<>();
        UniformVertex latentScale = new UniformVertex(0.01, 10.0);
        latentScale.setAndCascade(DoubleTensor.scalar(0.1));
        latentScaleList.add(latentScale);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new HalfCauchyVertex(new long[]{numSamples, 1}, hyperParams.get(0)),
            scale,
            latentScaleList,
            random
        );
    }

    @Test
    public void outOfBoundsGradientCalculation() {
        HalfCauchyVertex cauchyVertex = new HalfCauchyVertex(new long[]{2}, 5);
        DoubleTensor value = DoubleTensor.create(-5.0, 5.0);
        Map<Vertex, DoubleTensor> actualDerivatives = cauchyVertex.dLogPdf(value, cauchyVertex);
        DoubleTensor derivative = actualDerivatives.get(cauchyVertex);
        Assert.assertEquals(0.0, derivative.getValue(0), 1e-6);
        Assert.assertTrue(derivative.getValue(1) != 0.);
    }
}
