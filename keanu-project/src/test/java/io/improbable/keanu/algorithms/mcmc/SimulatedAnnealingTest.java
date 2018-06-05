package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.TensorGradientOptimizer;
import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.network.SimpleNetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class SimulatedAnnealingTest {

    private KeanuRandom random;
    private DoubleTensorVertex A;
    private DoubleTensorVertex B;
    private DoubleTensorVertex C;
    private DoubleTensorVertex D;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        A = new TensorGaussianVertex(5, 1);
        A.setValue(5.0);
        B = new TensorGaussianVertex(2, 1);
        B.setValue(2.0);
        C = A.plus(B);
        D = new TensorGaussianVertex(C, 1);
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
        BayesNetTensorAsContinuous network = new BayesNetTensorAsContinuous(A.getConnectedGraph());
        network.probeForNonZeroMasterP(100, random);

        TensorGradientOptimizer graphOptimizer = new TensorGradientOptimizer(network);
        graphOptimizer.maxAPosteriori(1000);

        return new SimpleNetworkState(network.getLatentVertices().stream()
            .collect(Collectors.toMap(Vertex::getId, Vertex::getValue)));
    }
}
