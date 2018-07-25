package io.improbable.keanu.algorithms.mcmc.initialconditions;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class MultimodalSimulatedAnnealingTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void findsBothModesForContinuousNetwork() {

        DoubleVertex A = VertexOfType.uniform(-3.0, 3.0);
        A.setValue(0.0);
        DoubleVertex B = A.multiply(A);
        DoubleVertex C = VertexOfType.gaussian(B, ConstantVertex.of(1.5));
        C.observe(DoubleTensor.scalar(4.0));

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 100, 1000, random);

        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() + 2) < 0.01);
        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() - 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

    @Test
    public void findsModesForDiscreteContinuousHybridNetwork() {

        UniformVertex A = VertexOfType.uniform(0.0, 3.0);
        A.setValue(1.0);
        DoubleVertex B = A.multiply(A);

        DoubleVertex C = VertexOfType.uniform(-3.0, 0.0);
        DoubleVertex D = C.multiply(C);

        BooleanVertex E = new Flip(0.5);

        DoubleVertex F = If.isTrue(E)
            .then(B)
            .orElse(D);

        DoubleVertex G = VertexOfType.gaussian(new CastDoubleVertex(F), ConstantVertex.of(1.5));
        G.observe(DoubleTensor.scalar(4.0));

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 100, 1000, random);

        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() - 2) < 0.01);
        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(C).scalar() + 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

}
