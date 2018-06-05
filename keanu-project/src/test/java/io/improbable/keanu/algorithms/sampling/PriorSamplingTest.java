package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class PriorSamplingTest {

    private DoubleTensorVertex A;
    private DoubleTensorVertex B;
    private DoubleTensorVertex C;
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        A = new TensorGaussianVertex(100.0, 1);
        B = new TensorGaussianVertex(A, 1);
        C = new TensorGaussianVertex(B, 1);
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

        B.observe(95.0);
        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());

        final int sampleCount = 10000;
        Prior.sample(net, net.getLatentVertices(), sampleCount, random);
    }

}
