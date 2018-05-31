package io.improbable.keanu.algorithms.mcmc.initialconditions;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.booltensor.BoolVertex;
import io.improbable.keanu.vertices.booltensor.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.CastDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.DoubleIfVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.If;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

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

    @Test
    public void findsBothModesForContinuousNetwork() {

        DoubleVertex A = new UniformVertex(-3.0, 3.0);
        A.setValue(0.0);
        DoubleVertex B = A.multiply(A);
        DoubleVertex C = new GaussianVertex(B, 1.5);
        C.observe(4.0);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 100, 1000, random);

        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(A) + 2) < 0.01);
        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A) - 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

    @Test
    public void findsModesForDiscreteContinuousHybridNetwork() {

        TensorUniformVertex A = new TensorUniformVertex(0.0, 3.0);
        A.setValue(1.0);
        DoubleTensorVertex B = A.multiply(A);

        DoubleTensorVertex C = new TensorUniformVertex(-3.0, 0.0);
        DoubleTensorVertex D = C.multiply(C);

        BoolVertex E = new Flip(0.5);

        DoubleIfVertex F = If.isTrue(E)
            .then(B)
            .orElse(D);

        DoubleTensorVertex G = new TensorGaussianVertex(new CastDoubleTensorVertex(F), 1.5);
        G.observe(4.0);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        List<NetworkState> modes = MultiModeDiscovery.findModesBySimulatedAnnealing(network, 100, 1000, random);

        boolean findsUpperMode = modes.stream().anyMatch(state -> Math.abs(state.get(A).scalar() - 2) < 0.01);
        boolean findsLowerMode = modes.stream().anyMatch(state -> Math.abs(state.get(C).scalar() + 2) < 0.01);

        assertTrue(findsLowerMode);
        assertTrue(findsUpperMode);
    }

}
