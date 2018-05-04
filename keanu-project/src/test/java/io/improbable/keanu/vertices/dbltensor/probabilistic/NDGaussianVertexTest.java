package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class NDGaussianVertexTest {

    private KeanuRandom random;

//    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

//    @Test
    public void inferHyperParamsFromSamples() {

        double trueMu = 4.5;
        double trueSigma = 2.0;

        List<DoubleTensorVertex> muSigma = new ArrayList<>();
        muSigma.add(new ConstantVertex(DoubleTensor.nd4JScalar(trueMu)));
        muSigma.add(new ConstantVertex(DoubleTensor.nd4JScalar(trueSigma)));

        List<DoubleTensorVertex> latentMuSigma = new ArrayList<>();
        latentMuSigma.add(new NDUniformVertex(0.01, 10.0, random));
        latentMuSigma.add(new NDUniformVertex(0.01, 10.0, random));

        int numSamples = 10000;
        TensorVertexVariationalMAP.inferHyperParamsFromSamples(
                hyperParams -> new NDGaussianVertex(new int[]{numSamples, 1}, hyperParams.get(0), hyperParams.get(1), random),
                muSigma,
                latentMuSigma,
                numSamples
        );
    }
}
