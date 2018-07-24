package io.improbable.keanu.vertices.dbl.probabilistic;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class TriangularVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        TriangularVertex tensorTriangularVertex = VertexOfType.triangular(0.0, 10., 5.);
        double expectedLogDensity = Math.log(0.1);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorTriangularVertex, 2.5, expectedLogDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        TriangularVertex triangularVertex = VertexOfType.triangular(0.0, 10., 5.);

        double expectedLogDensity = triangularVertex.logProb(DoubleTensor.scalar(2.5)) + triangularVertex.logProb(DoubleTensor.scalar(7.5));
        TriangularVertex tensorTriangularVertex = VertexOfType.triangular(0.0, 10., 5.);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorTriangularVertex, new double[]{2.5, 7.5}, expectedLogDensity);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        TriangularVertex vertex = new DistributionVertexBuilder()
            .shaped(sampleCount, 1)
            .withInput(ParameterName.MIN, 0.0)
            .withInput(ParameterName.MAX, 10.0)
            .withInput(ParameterName.C, 5.0)
            .triangular();

        double from = 1;
        double to = 9;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }
}
