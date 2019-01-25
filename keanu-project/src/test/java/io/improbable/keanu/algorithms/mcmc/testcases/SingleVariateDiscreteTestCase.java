package io.improbable.keanu.algorithms.mcmc.testcases;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class SingleVariateDiscreteTestCase implements MCMCTestCase {

    private final BayesianNetwork model;

    private final BernoulliVertex A;

    public SingleVariateDiscreteTestCase() {
        A = new BernoulliVertex(0.5);

        DoubleVertex B = If.isTrue(A)
            .then(0.9)
            .orElse(0.1);

        BernoulliVertex C = new BernoulliVertex(B);

        C.observe(true);

        model = new BayesianNetwork(Arrays.asList(A, B, C));
    }

    @Override
    public BayesianNetwork getModel() {
        return model;
    }

    @Override
    public void assertExpected(NetworkSamples posteriorSamples) {
        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.9, postProbTrue, 0.01);
    }
}
