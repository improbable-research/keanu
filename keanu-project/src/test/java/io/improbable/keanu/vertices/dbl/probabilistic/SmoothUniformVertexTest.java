package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import umontreal.ssj.probdist.UniformDist;

import java.util.Map;

import static org.junit.Assert.assertEquals;

public class SmoothUniformVertexTest {

    private static final double DELTA = 0.0001;

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        SmoothUniformVertex tensorSmoothUniformVertex = new SmoothUniformVertex(0, 1);
        SmoothUniformVertex smoothUniformVertex = new SmoothUniformVertex(0, 1);
        double expectedDensity = smoothUniformVertex.logPdf(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorSmoothUniformVertex, 0.5, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesLogDensityOfScalar() {
        DoubleVertex xMin = ConstantVertex.of(0.);
        DoubleVertex xMax = ConstantVertex.of(100.);
        SmoothUniformVertex smoothUniformVertex = new SmoothUniformVertex(xMin, xMax, 1e-6);
        LogProbGraph logProbGraph = smoothUniformVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, smoothUniformVertex, DoubleTensor.scalar(50.));

        UniformDist uniformDist = new UniformDist(0., 100.);
        double expectedDensity = Math.log(uniformDist.density(50.));

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfVector() {

        SmoothUniformVertex smoothUniformVertex = new SmoothUniformVertex(0, 1);
        double expectedLogDensity = smoothUniformVertex.logPdf(0.25) + smoothUniformVertex.logPdf(0.75);
        SmoothUniformVertex tensorSmoothUniformVertex = new SmoothUniformVertex(0, 1);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorSmoothUniformVertex,
            new double[]{0.25, 0.75},
            expectedLogDensity
        );
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex xMin = ConstantVertex.of(0., 0.);
        DoubleVertex xMax = ConstantVertex.of(100., 100.);
        SmoothUniformVertex smoothUniformVertex = new SmoothUniformVertex(xMin, xMax, 1e-6);
        LogProbGraph logProbGraph = smoothUniformVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, smoothUniformVertex, DoubleTensor.create(25., 75.));

        UniformDist uniformDist = new UniformDist(0., 100.);
        double expectedDensity = Math.log(uniformDist.density(25.) * uniformDist.density(75.));

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {

        SmoothUniformVertex smoothUniformVertex = new SmoothUniformVertex(0, 1, 10);
        SmoothUniformVertex tensorSmoothUniformVertex = new SmoothUniformVertex(0, 1, 10);

        Map<Vertex, DoubleTensor> derivativeFlatRegion = smoothUniformVertex.dLogPdf(0.5, smoothUniformVertex);
        Map<Vertex, DoubleTensor> tensorDerivativeFlatRegion = tensorSmoothUniformVertex.dLogPdf(0.5, tensorSmoothUniformVertex);

        assertEquals(derivativeFlatRegion.get(smoothUniformVertex).scalar(),
            tensorDerivativeFlatRegion.get(tensorSmoothUniformVertex).scalar(),
            DELTA
        );

        Map<Vertex, DoubleTensor> derivativeLeftRegion = smoothUniformVertex.dLogPdf(-0.5, smoothUniformVertex);
        Map<Vertex, DoubleTensor> tensorDerivativeLeftRegion = tensorSmoothUniformVertex.dLogPdf(-0.5, tensorSmoothUniformVertex);

        assertEquals(derivativeLeftRegion.get(smoothUniformVertex).scalar(),
            tensorDerivativeLeftRegion.get(tensorSmoothUniformVertex).scalar(),
            DELTA
        );

        Map<Vertex, DoubleTensor> derivativeRightRegion = smoothUniformVertex.dLogPdf(1.5, smoothUniformVertex);
        Map<Vertex, DoubleTensor> tensorDerivativeRightRegion = tensorSmoothUniformVertex.dLogPdf(1.5, tensorSmoothUniformVertex);

        assertEquals(derivativeRightRegion.get(smoothUniformVertex).scalar(),
            tensorDerivativeRightRegion.get(tensorSmoothUniformVertex).scalar(),
            DELTA
        );
    }

    @Category(Slow.class)
    @Test
    public void smoothUniformSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;

        double edgeSharpness = 1.0;
        SmoothUniformVertex vertex = new SmoothUniformVertex(
            new long[]{sampleCount, 1},
            0.0,
            1.0,
            edgeSharpness
        );

        double from = -1;
        double to = 2;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex,
            from,
            to,
            bucketSize,
            1e-2,
            random
        );
    }

}
