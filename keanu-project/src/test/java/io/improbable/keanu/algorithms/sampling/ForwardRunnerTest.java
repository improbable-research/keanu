package io.improbable.keanu.algorithms.sampling;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.backend.KeanuComputableGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class ForwardRunnerTest {

    private DoubleVertex A;
    private DoubleVertex B;
    private DoubleVertex C;
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        A = new GaussianVertex(100.0, 1);
        B = new GaussianVertex(A, 1);
        C = new GaussianVertex(B, 1);
    }

    @Test
    public void samplesFromPrior() {
        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());
        KeanuComputableGraph graph = new KeanuComputableGraph(C.getConnectedGraph());

        final int sampleCount = 1000;
        NetworkSamples samples = ForwardRunner.sample(graph, net.getLatentVertices(), sampleCount, random);

        double averageC = samples.getDoubleTensorSamples(C).getAverages().scalar();

        assertEquals(sampleCount, samples.size());
        assertEquals(100.0, averageC, 0.1);
    }

}
