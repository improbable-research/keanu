package io.improbable.keanu.algorithms.mcmc.testcases;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class MultiVariateDiscreteTestCase implements MCMCTestCase {

    private final BayesianNetwork model;

    private final BernoulliVertex A;
    private final BernoulliVertex B;

    public MultiVariateDiscreteTestCase() {
        A = new BernoulliVertex(0.5);
        B = new BernoulliVertex(0.5);

        BooleanVertex C = A.or(B);

        DoubleVertex D = If.isTrue(C)
            .then(0.9)
            .orElse(0.1);

        BernoulliVertex E = new BernoulliVertex(D);

        E.observe(true);

        model = new BayesianNetwork(Arrays.asList(A, B, C, D, E));
    }

    @Override
    public BayesianNetwork getModel() {
        return model;
    }

    @Override
    public void assertExpected(NetworkSamples posteriorSamples) {
        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.643, postProbTrue, 0.01);
    }
}
