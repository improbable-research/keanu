package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbl.probabilistic.TriangularVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import org.junit.Before;
import org.junit.Test;

public class TensorTriangularVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {

        TensorTriangularVertex tensorTriangularVertex = new TensorTriangularVertex(0.0, 10., 5.);
        TriangularVertex triangularVertex = new TriangularVertex(0.0, 10, 5);
        double expectedDensity = triangularVertex.logPdf(2.5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorTriangularVertex, 2.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        TriangularVertex triangularVertex = new TriangularVertex(0.0, 10, 5);

        double expectedLogDensity = triangularVertex.logPdf(2.5) + triangularVertex.logPdf(7.5);
        TensorTriangularVertex tensorTriangularVertex = new TensorTriangularVertex(0.0, 10, 5);
        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorTriangularVertex, new double[]{2.5, 7.5}, expectedLogDensity);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {

        int sampleCount = 1000000;
        TensorTriangularVertex vertex = new TensorTriangularVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(0.0),
            new ConstantTensorVertex(10.0),
            new ConstantTensorVertex(5.0)
        );

        double from = 1;
        double to = 9;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2, random);
    }
}
