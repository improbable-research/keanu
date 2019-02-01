package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;
import umontreal.ssj.probdist.TriangularDist;

public class TriangularVertexTest {

    private KeanuRandom random;

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbMatchesKnownLogDensityOfScalar() {

        TriangularVertex tensorTriangularVertex = new TriangularVertex(0.0, 10., 5.);
        double expectedLogDensity = Math.log(0.1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorTriangularVertex, 2.5, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        DoubleVertex xMin = ConstantVertex.of(0.);
        DoubleVertex xMax = ConstantVertex.of(20.);
        DoubleVertex c = ConstantVertex.of(10.);
        TriangularVertex triangularVertex = new TriangularVertex(xMin, xMax, c);
        LogProbGraph logProbGraph = triangularVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, c, c.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, triangularVertex, DoubleTensor.scalar(2.5));

        TriangularDist triangularDist = new TriangularDist(0., 20., 10.);
        double expectedDensity = Math.log(triangularDist.density(2.5));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        TriangularVertex triangularVertex = new TriangularVertex(0.0, 10, 5);

        double expectedLogDensity = triangularVertex.logPdf(2.5) + triangularVertex.logPdf(7.5);
        TriangularVertex tensorTriangularVertex = new TriangularVertex(0.0, 10, 5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorTriangularVertex, new double[]{2.5, 7.5}, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        DoubleVertex xMin = ConstantVertex.of(0., 0.);
        DoubleVertex xMax = ConstantVertex.of(20., 20.);
        DoubleVertex c = ConstantVertex.of(10., 10.);
        TriangularVertex triangularVertex = new TriangularVertex(xMin, xMax, c);
        LogProbGraph logProbGraph = triangularVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, c, c.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, triangularVertex, DoubleTensor.create(2.5, 7.5));

        TriangularDist triangularDist = new TriangularDist(0., 20., 10.);
        double expectedDensity = Math.log(triangularDist.density(2.5) * triangularDist.density(7.5));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void cLessThanXMinThrowsException() {
        DoubleVertex xMin = ConstantVertex.of(0.);
        DoubleVertex xMax = ConstantVertex.of(1.);
        DoubleVertex c = ConstantVertex.of(-1.);
        TriangularVertex triangularVertex = new TriangularVertex(xMin, xMax, c);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("center must be between xMin and xMax. c: " + c.getValue() + " xMin: " + xMin.getValue() + " xMax: " + xMax.getValue());
        triangularVertex.sample();
    }

    @Test
    public void cGreaterThanXMaxThrowsException() {
        DoubleVertex xMin = ConstantVertex.of(0.);
        DoubleVertex xMax = ConstantVertex.of(1.);
        DoubleVertex c = ConstantVertex.of(2.);
        TriangularVertex triangularVertex = new TriangularVertex(xMin, xMax, c);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("center must be between xMin and xMax. c: " + c.getValue() + " xMin: " + xMin.getValue() + " xMax: " + xMax.getValue());
        triangularVertex.sample();
    }

    @Test
    public void cEqualToXMinOrXMaxDoesNotThrowException() {
        DoubleVertex xMin = ConstantVertex.of(0., 0.);
        DoubleVertex xMax = ConstantVertex.of(1., 1.);
        DoubleVertex c = ConstantVertex.of(0., 1.);
        TriangularVertex triangularVertex = new TriangularVertex(xMin, xMax, c);
        triangularVertex.sample();
    }

    @Category(Slow.class)
    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        TriangularVertex vertex = new TriangularVertex(
            new long[]{sampleCount, 1},
            0.0,
            10.0,
            5.0
        );

        double from = 1;
        double to = 9;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }
}
