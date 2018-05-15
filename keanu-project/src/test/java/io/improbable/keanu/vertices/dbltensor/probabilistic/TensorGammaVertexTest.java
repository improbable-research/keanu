package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import org.junit.Test;

public class TensorGammaVertexTest {

    @Test
    public void matchesKnownLogDensityOfScalar() {
        
    }

    @Test
    public void gammaSampledMethodMatchesLogProbMethod() {
        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        TensorGammaVertex vertex = new TensorGammaVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(1.5),
            new ConstantTensorVertex(2.0),
            new ConstantTensorVertex(2.5),
            random
        );

        double from = 1;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2);
    }

}
