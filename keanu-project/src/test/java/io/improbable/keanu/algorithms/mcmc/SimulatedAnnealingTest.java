package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

public class SimulatedAnnealingTest {

    private Random random;
    private DoubleVertex A;
    private DoubleVertex B;
    private DoubleVertex C;
    private DoubleVertex D;

    @Before
    public void setup() {
        random = new Random(1);
        A = new GaussianVertex(5, 1, random);
        B = new GaussianVertex(2, 1, random);
        C = A.plus(B);
        D = new GaussianVertex(C, 1, random);
        D.observe(7.5);
    }

    @Test
    public void findsMaxAposterioriWithAnnealing() {

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        network.probeForNonZeroMasterP(100);

        NetworkState maxAPosterioriSamples = SimulatedAnnealing.getMaxAPosteriori(network, 10000, random);
        Map<Long, ?> maxValuesFromVariational = findMAPWithOptimizer();

        for (long id : maxAPosterioriSamples.getVertexIds()) {
            assertEquals((Double) maxValuesFromVariational.get(id), maxAPosterioriSamples.get(id), 0.01);
        }
    }

    private Map<Long, ?> findMAPWithOptimizer() {
        BayesNetDoubleAsContinuous network = new BayesNetDoubleAsContinuous(A.getConnectedGraph());
        network.probeForNonZeroMasterP(100);

        GradientOptimizer graphOptimizer = new GradientOptimizer(network);
        graphOptimizer.maxAPosteriori(1000);

        return network.getLatentVertices().stream()
            .collect(Collectors.toMap(Vertex::getId, Vertex::getValue));
    }
}
