package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Collections;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class IntegerIfVertexTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void correctBranchTaken() {

        BooleanVertex predicate = new BernoulliVertex(0.5);

        IntegerVertex ifIsTrue = If.isTrue(predicate)
            .then(1)
            .orElse(0);

        predicate.setAndCascade(true);
        assertEquals(1, ifIsTrue.getValue().scalar().intValue());

        predicate.setAndCascade(false);
        assertEquals(0, ifIsTrue.getValue().scalar().intValue());
    }

    @Test
    public void expectedValueMatchesBranchProbability() {

        double p = 0.5;
        int thenValue = 2;
        int elseValue = 4;
        double expectedMean = p * thenValue + (1 - p) * elseValue;

        BernoulliVertex predicate = new BernoulliVertex(p);

        IntegerVertex vertex = If.isTrue(predicate)
            .then(thenValue)
            .orElse(elseValue);

        assertEquals(calculateMeanOfVertex(vertex), expectedMean, 0.01);
    }

    private static double calculateMeanOfVertex(IntegerVertex vertex) {
        BayesianNetwork net = new BayesianNetwork(vertex.getConnectedGraph());

        return MetropolisHastings.withDefaultConfig(KeanuRandom.getDefaultRandom())
            .generatePosteriorSamples(net, Collections.singletonList(vertex)).stream()
            .limit(2000)
            .collect(Collectors.averagingInt((NetworkSample state) -> state.get(vertex).scalar()));
    }
}
