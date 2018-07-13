package io.improbable.keanu.algorithms.sampling;

import static junit.framework.TestCase.assertEquals;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class PriorSamplingTest {

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
    public void samplesFromPriorWithMultiMarkovBlanketNetwork() {

        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());

        final int sampleCount = 10000;
        NetworkSamples samples = Prior.sample(net, net.getLatentVertices(), sampleCount, random);

        double averageC = samples.getDoubleTensorSamples(C).getAverages().scalar();

        assertEquals(sampleCount, samples.size());
        assertEquals(100.0, averageC, 0.1);
    }

    @Test(expected = IllegalStateException.class)
    public void doesNotSamplePriorFromNetWithObservations() {

        B.observe(DoubleTensor.scalar(95.0));
        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());

        final int sampleCount = 10000;
        Prior.sample(net, net.getLatentVertices(), sampleCount, random);
    }

}
