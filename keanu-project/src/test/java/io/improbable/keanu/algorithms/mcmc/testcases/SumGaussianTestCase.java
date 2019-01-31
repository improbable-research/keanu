package io.improbable.keanu.algorithms.mcmc.testcases;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class SumGaussianTestCase implements MCMCTestCase {

    private final DoubleVertex A;
    private final DoubleVertex B;

    private BayesianNetwork model;

    public SumGaussianTestCase() {
        A = new GaussianVertex(20.0, 1.0);
        B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        model = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        model.probeForNonZeroProbability(100);
    }

    @Override
    public BayesianNetwork getModel() {
        return model;
    }

    @Override
    public void assertExpected(NetworkSamples posteriorSamples) {

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar();

        double actual = averagePosteriorA + averagePosteriorB;
        assertEquals(44.0, actual, 0.1);
    }
}
