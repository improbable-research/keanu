package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Cauchy;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.CauchyDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class CauchyVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        CauchyVertex tensorCauchyVertex = new CauchyVertex(0, 1);
        double expectedDensity = distribution.logDensity(0.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorCauchyVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex location = ConstantVertex.of(0.);
        DoubleVertex scale = ConstantVertex.of(1.);
        CauchyVertex vertex = new CauchyVertex(location, scale);
        LogProbGraph logProbGraph = vertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, location, location.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.scalar(0.5));

        CauchyDistribution distribution = new CauchyDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        CauchyDistribution distribution = new CauchyDistribution(0.0, 1.0);
        double expectedLogDensity = distribution.logDensity(0.25) + distribution.logDensity(-0.75);
        CauchyVertex tensorCauchyVertex = new CauchyVertex(0, 1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorCauchyVertex, new double[]{0.25, -0.75}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownDensityOfVector() {
        DoubleVertex location = ConstantVertex.of(0., 0.);
        DoubleVertex scale = ConstantVertex.of(1., 1.);
        CauchyVertex vertex = new CauchyVertex(location, scale);
        LogProbGraph logProbGraph = vertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, location, location.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.create(0.25, -0.75));

        CauchyDistribution distribution = new CauchyDistribution(0., 1.);
        double expectedDensity = distribution.logDensity(0.25) + distribution.logDensity(-0.75);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        Cauchy.Diff cauchyLogDiff = Cauchy.dlnPdf(0.0, 1.0, 0.5);

        UniformVertex locationTensor = new UniformVertex(0.0, 1.0);
        locationTensor.setValue(0.0);

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        CauchyVertex tensorCauchyVertex = new CauchyVertex(locationTensor, scaleTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = tensorCauchyVertex.dLogPdf(0.5, locationTensor, scaleTensor, tensorCauchyVertex);

        assertEquals(cauchyLogDiff.dPdlocation, actualDerivatives.get(locationTensor).scalar(), 1e-5);
        assertEquals(cauchyLogDiff.dPdscale, actualDerivatives.get(scaleTensor).scalar(), 1e-5);
        assertEquals(cauchyLogDiff.dPdx, actualDerivatives.get(tensorCauchyVertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{0.25, -0.75, 0.1, -2, 1.3};

        UniformVertex locationTensor = new UniformVertex(0.0, 1.0);
        locationTensor.setValue(0.0);

        UniformVertex scaleTensor = new UniformVertex(0.0, 1.0);
        scaleTensor.setValue(1.0);

        Supplier<CauchyVertex> vertexSupplier = () -> new CauchyVertex(locationTensor, scaleTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex location = new UniformVertex(0.0, 1.0);
        location.setAndCascade(Nd4jDoubleTensor.scalar(0.5));
        CauchyVertex vertexUnderTest = new CauchyVertex(
            location,
            3.0
        );
        vertexUnderTest.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdlocation() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        CauchyVertex cauchy = new CauchyVertex(uniformA, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(0.0);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);
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

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdscale() {
        UniformVertex uniformA = new UniformVertex(1.5, 3.0);
        CauchyVertex cauchy = new CauchyVertex(3.0, uniformA);

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
        CauchyVertex vertex = new CauchyVertex(
            new long[]{sampleCount, 1},
            ConstantVertex.of(0.0),
            ConstantVertex.of(2.0)
        );

        double from = -4;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueLocation = 4.5;
        double trueScale = 2.0;

        List<DoubleVertex> locationScale = new ArrayList<>();
        locationScale.add(ConstantVertex.of(trueLocation));
        locationScale.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentLocationScale = new ArrayList<>();
        UniformVertex latentLocation = new UniformVertex(0.01, 10.0);
        latentLocation.setAndCascade(DoubleTensor.scalar(9.9));
        UniformVertex latentScale = new UniformVertex(0.01, 10.0);
        latentScale.setAndCascade(DoubleTensor.scalar(0.1));
        latentLocationScale.add(latentLocation);
        latentLocationScale.add(latentScale);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new CauchyVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            locationScale,
            latentLocationScale,
            random
        );
    }
}
