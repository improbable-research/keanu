package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class SimulatedAnnealingTest {

    private KeanuRandom random;
    private DoubleVertex A;
    private DoubleVertex B;
    private DoubleVertex C;
    private DoubleVertex D;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        A = new GaussianVertex(5, 1);
        A.setValue(5.0);
        B = new GaussianVertex(2, 1);
        B.setValue(2.0);
        C = A.plus(B);
        D = new GaussianVertex(C, 1);
        D.observe(7.5);
    }

    @Test
    public void findsMaxAposterioriWithAnnealing() {

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        network.probeForNonZeroMasterP(100, random);

        NetworkState maxAPosterioriSamples = SimulatedAnnealing.getMaxAPosteriori(network, 10000, random);
        NetworkState maxValuesFromVariational = findMAPWithOptimizer();

        assertEquals(maxValuesFromVariational.get(A).scalar(), maxAPosterioriSamples.get(A).scalar(), 0.05);
        assertEquals(maxValuesFromVariational.get(B).scalar(), maxAPosterioriSamples.get(B).scalar(), 0.05);
    }

    private NetworkState findMAPWithOptimizer() {
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        network.probeForNonZeroMasterP(100, random);

        GradientOptimizer graphOptimizer = new GradientOptimizer(network);
        graphOptimizer.maxAPosteriori();

        return new SimpleNetworkState(network.getLatentVertices().stream()
            .collect(Collectors.toMap(Vertex::getId, Vertex::getValue)));
    }
}
