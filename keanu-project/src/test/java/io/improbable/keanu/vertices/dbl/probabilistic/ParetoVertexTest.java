package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.gradient.Pareto;
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
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.utility.GraphAssertionException;
import org.apache.commons.math3.distribution.ParetoDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static org.junit.Assert.assertEquals;

public class ParetoVertexTest {

    private final double VERTEX_INCREMENT = 0.1;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {
        ParetoDistribution baseline = new ParetoDistribution(1.0, 1.5);
        ParetoVertex vertex = new ParetoVertex(1.0, 1.5);
        double expected = baseline.logDensity(1.25);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(vertex, 1.25, expected);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex location = ConstantVertex.of(1.);
        DoubleVertex scale = ConstantVertex.of(1.5);
        ParetoVertex paretoVertex = new ParetoVertex(location, scale);
        LogProbGraph logProbGraph = paretoVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, location, location.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, paretoVertex, DoubleTensor.scalar(1.25));

        ParetoDistribution baseline = new ParetoDistribution(1., 1.5);
        double expected = baseline.logDensity(1.25);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {
        ParetoDistribution baseline = new ParetoDistribution(1.0, 1.5);
        ParetoVertex vertex = new ParetoVertex(1.0, 1.5);
        double expected = baseline.logDensity(1.25) + baseline.logDensity(6.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(vertex, new double[]{1.25, 6.5}, expected);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex location = ConstantVertex.of(1., 1.);
        DoubleVertex scale = ConstantVertex.of(1.5, 1.5);
        ParetoVertex paretoVertex = new ParetoVertex(location, scale);
        LogProbGraph logProbGraph = paretoVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, location, location.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, paretoVertex, DoubleTensor.create(1.25, 6.5));

        ParetoDistribution baseline = new ParetoDistribution(1., 1.5);
        double expected = baseline.logDensity(1.25) + baseline.logDensity(6.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Test(expected = GraphAssertionException.class)
    public void logProbGraphIsNegInfIfLocationOrScaleIsNotPositive() {
        DoubleVertex location = ConstantVertex.of(-1., 1.);
        DoubleVertex scale = ConstantVertex.of(0., 3.);
        ParetoVertex paretoVertex = new ParetoVertex(location, scale);
        LogProbGraph logProbGraph = paretoVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, paretoVertex, DoubleTensor.create(2., 2.));
        LogProbGraphValueFeeder.feedValueAndCascade(logProbGraph, location, location.getValue());
    }

    @Test
    public void logProbGraphIsNegInfIfXIsLessThanOrEqualToLocation() {
        DoubleVertex location = ConstantVertex.of(1., 1.);
        DoubleVertex scale = ConstantVertex.of(10., 10.);
        ParetoVertex paretoVertex = new ParetoVertex(location, scale);
        LogProbGraph logProbGraph = paretoVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, location, location.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, scale, scale.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, paretoVertex, DoubleTensor.create(2., .5));

        ParetoDistribution distribution = new ParetoDistribution(1., 10.);
        double expectedDensity = distribution.logDensity(2.);

        LogProbGraphContract.equalFlatArray(logProbGraph, new double[] {expectedDensity, Double.NEGATIVE_INFINITY});
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        Pareto.Diff paretoLogDiff = Pareto.dlnPdf(1.0, 1.5, 2.5);

        UniformVertex locationTensor = new UniformVertex(0.0, 1.0);
        locationTensor.setValue(1.0);

        UniformVertex scaleTensor = new UniformVertex(0.0, 2);
        scaleTensor.setValue(1.5);

        ParetoVertex vertex = new ParetoVertex(locationTensor, scaleTensor);
        Map<Vertex, DoubleTensor> actualDerivatives = vertex.dLogPdf(2.5, locationTensor, scaleTensor, vertex);

        assertEquals(paretoLogDiff.dPdLocation, actualDerivatives.get(locationTensor).scalar(), 1e-5);
        assertEquals(paretoLogDiff.dPdScale, actualDerivatives.get(scaleTensor).scalar(), 1e-5);
        assertEquals(paretoLogDiff.dPdX, actualDerivatives.get(vertex).scalar(), 1e-5);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfVector() {

        double[] vector = new double[]{1.1, 1.3, 1.8, 2.5, 5};

        UniformVertex locationTensor = new UniformVertex(0.0, 1.0);
        locationTensor.setValue(1.0);

        UniformVertex scaleTensor = new UniformVertex(0.0, 2.0);
        scaleTensor.setValue(1.5);

        Supplier<ParetoVertex> vertexSupplier = () -> new ParetoVertex(locationTensor, scaleTensor);

        ProbabilisticDoubleTensorContract.matchesKnownDerivativeLogDensityOfVector(vector, vertexSupplier);
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        UniformVertex xm = new UniformVertex(0.0, 1.0);
        xm.setAndCascade(Nd4jDoubleTensor.scalar(1.0));
        ParetoVertex vertexUnderTest = new ParetoVertex(xm, 3.0);
        vertexUnderTest.setAndCascade(1.0);
        ProbabilisticDoubleTensorContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleTensorContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdLocation() {
        UniformVertex location = new UniformVertex(0.1, 1.0);
        ParetoVertex pareto = new ParetoVertex(location, 3.0);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.5);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.1),
            Nd4jDoubleTensor.scalar(1.0),
            0.1,
            location,
            pareto,
            vertexStartValue,
            vertexEndValue,
            VERTEX_INCREMENT,
            1e-5
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdScale() {
        UniformVertex scale = new UniformVertex(0.1, 5.0);
        ParetoVertex pareto = new ParetoVertex(1.0, scale);

        DoubleTensor vertexStartValue = Nd4jDoubleTensor.scalar(1.5);
        DoubleTensor vertexEndValue = Nd4jDoubleTensor.scalar(5.0);

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            Nd4jDoubleTensor.scalar(0.1),
            Nd4jDoubleTensor.scalar(5.0),
            0.1,
            scale,
            pareto,
            vertexStartValue,
            vertexEndValue,
            VERTEX_INCREMENT,
            1e-5
        );
    }

    @Category(Slow.class)
    @Test
    public void sampleMatchesLogProb() {
        int sampleCount = 1000000;
        ParetoVertex vertex = new ParetoVertex(new long[]{sampleCount, 1}, 1.0, 3.0);

        double from = 1.0;
        double to = 2.5;
        double bucketSize = 0.01;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 5e-2,
            random);
    }

    @Test
    public void inferHyperParamsFromSamples() {
        double trueLocation = 5.0;

        /*
         * Note, this value is set low as the Gradient Optimizer seems to struggle with values greater than this - we
         * should revisit with a more configurable Optimiser or a different Optimiser Entirely to see if we can get this
         * to work for all scale values.
         */
        double trueScale = 0.1;

        List<DoubleVertex> trueParams = new ArrayList<>();
        trueParams.add(ConstantVertex.of(trueLocation));
        trueParams.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentParams = new ArrayList<>();
        UniformVertex latentLocation = new UniformVertex(0.1, 15.0);
        latentLocation.setAndCascade(6.0);
        UniformVertex latentScale = new UniformVertex(0.01, 10);
        latentScale.setAndCascade(9.0);
        latentParams.add(latentLocation);
        latentParams.add(latentScale);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new ParetoVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            trueParams,
            latentParams,
            random
        );
    }

    @Test
    public void inferHyperParamsFromSamplesFixedLocation() {
        double trueLocation = 5.0;
        double trueScale = 3.5;

        List<DoubleVertex> trueParams = new ArrayList<>();
        trueParams.add(ConstantVertex.of(trueLocation));
        trueParams.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentParams = new ArrayList<>();
        ConstantDoubleVertex latentLocation = new ConstantDoubleVertex(trueLocation);
        UniformVertex latentScale = new UniformVertex(0.01, 10);
        latentScale.setAndCascade(0.1);
        latentParams.add(latentLocation);
        latentParams.add(latentScale);

        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new ParetoVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            trueParams,
            latentParams,
            random
        );
    }

    @Test
    public void inferHyperParamsFromSamplesFixedScale() {
        double trueLocation = 5.0;
        double trueScale = 3.5;

        List<DoubleVertex> trueParams = new ArrayList<>();
        trueParams.add(ConstantVertex.of(trueLocation));
        trueParams.add(ConstantVertex.of(trueScale));

        List<DoubleVertex> latentParams = new ArrayList<>();
        UniformVertex latentLocation = new UniformVertex(0.01, 10);
        latentLocation.setAndCascade(10.0);
        ConstantDoubleVertex latentScale = new ConstantDoubleVertex(trueScale);
        latentParams.add(latentLocation);
        latentParams.add(latentScale);


        int numSamples = 2000;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new ParetoVertex(new long[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1)),
            trueParams,
            latentParams,
            random
        );
    }
}
