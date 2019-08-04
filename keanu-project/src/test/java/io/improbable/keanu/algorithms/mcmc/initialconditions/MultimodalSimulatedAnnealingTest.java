package io.improbable.keanu.algorithms.mcmc.initialconditions;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.List;

import static org.junit.Assert.assertTrue;

public class MultimodalSimulatedAnnealingTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Category(Slow.class)
    @Test
    public void findsBothModesForContinuousNetwork() {

        DoubleVertex A = new UniformVertex(-3.0, 3.0);
        A.setValue(0.0);
        DoubleVertex B = A.multiply(A);
        DoubleVertex C = new GaussianVertex(B, 1.5);
        C.observe(4.0);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 30, 1000, random);

        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() + 2) < 0.01);
        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() - 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

    @Category(Slow.class)
    @Test
    public void findsModesForDiscreteContinuousHybridNetwork() {

        UniformVertex A = new UniformVertex(0.0, 3.0);
        A.setValue(1.0);
        DoubleVertex B = A.multiply(A);

        DoubleVertex C = new UniformVertex(-3.0, 0.0);
        DoubleVertex D = C.multiply(C);

        BooleanVertex E = new BernoulliVertex(0.5);

        DoubleVertex F = If.isTrue(E)
            .then(B)
            .orElse(D);

        DoubleVertex G = new GaussianVertex(F, 1.5);
        G.observe(4.0);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 30, 1000, random);

        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() - 2) < 0.01);
        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(C).scalar() + 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

}
