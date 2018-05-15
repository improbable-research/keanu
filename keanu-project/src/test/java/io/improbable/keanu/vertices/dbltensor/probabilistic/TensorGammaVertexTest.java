package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import org.junit.Test;

import java.util.Random;

public class TensorGammaVertexTest {

    @Test
    public void matchesKnownLogDensityOfScalar() {
        GammaVertex gamma = new GammaVertex(0.5, 1, 1.5, new Random(1));
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.5, 1, 1.5, new KeanuRandom(1));

        double expectedDensity = gamma.logPdf(0.5);

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfScalar(tensorGamma, 0.5, expectedDensity);
    }

    @Test
    public void matchesKnownLogDensityOfVector() {
        Random random = new Random();

        ConstantDoubleVertex a = new ConstantDoubleVertex(0.5);
        ConstantDoubleVertex theta = new ConstantDoubleVertex(1.);
        ConstantDoubleVertex k = new ConstantDoubleVertex(7.);
        GammaVertex gammaA = new GammaVertex(a, theta, k, random);
        GammaVertex gammaB = new GammaVertex(a, theta, k, random);

        double expectedLogDensity = gammaA.logPdf(1.) + gammaB.logPdf(3.);
        TensorGammaVertex tensorGamma = new TensorGammaVertex(0.5, 1., 7., new KeanuRandom(1));

        ProbabilisticDoubleTensorContract.matchesKnownLogDensityOfVector(tensorGamma, new double[]{1., 3.}, expectedLogDensity);
    }

    @Test
    public void gammaSampledMethodMatchesLogProbMethod() {
        KeanuRandom random = new KeanuRandom(1);

        int sampleCount = 1000000;
        TensorGammaVertex vertex = new TensorGammaVertex(
            new int[]{sampleCount, 1},
            new ConstantTensorVertex(1.5),
            new ConstantTensorVertex(2.0),
            new ConstantTensorVertex(7.5),
            random
        );

        double from = 1.5;
        double to = 2.5;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(vertex, from, to, bucketSize, 1e-2);
    }

}
