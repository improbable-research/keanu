package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

public class TriangularVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        TriangularVertex tensorTriangularVertex = new TriangularVertex(0.0, 10., 5.);
        double expectedLogDensity = Math.log(0.1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorTriangularVertex, 2.5, expectedLogDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        TriangularVertex triangularVertex = new TriangularVertex(0.0, 10, 5);

        double expectedLogDensity = triangularVertex.logPdf(2.5) + triangularVertex.logPdf(7.5);
        TriangularVertex tensorTriangularVertex = new TriangularVertex(0.0, 10, 5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorTriangularVertex, new double[]{2.5, 7.5}, expectedLogDensity);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        TriangularVertex vertex = new TriangularVertex(
            new int[]{sampleCount, 1},
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
