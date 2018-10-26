package io.improbable.keanu.backend.keanu;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class KeanuProbabilisticGraphTest {

    GaussianVertex A;
    GaussianVertex B;

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0);
        A.setLabel("A");
        B = new GaussianVertex(0.0, 1.0);
        B.setLabel("B");
    }

    @Test
    public void canConvertSimpleBayesianNetwork() {
        DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
        C.observe(6.0);

        KeanuProbabilisticGraph probabilisticGraph = KeanuGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        double logProb = probabilisticGraph.logProb(ImmutableMap.of(
            "A", DoubleTensor.scalar(2),
            "B", DoubleTensor.scalar(3)
        ));

        NormalDistribution latents = new NormalDistribution(0.0, 1.0);
        NormalDistribution observation = new NormalDistribution(5.0, 1.0);
        double expected = latents.logDensity(2.0) + latents.logDensity(3.0) + observation.logDensity(6.0);
        assertEquals(expected, logProb, 1e-5);

        double logProb2 = probabilisticGraph.logProb(ImmutableMap.of(
            "A", DoubleTensor.scalar(3)
        ));

        double expected2 = latents.logDensity(3.0) + latents.logDensity(3.0) + new NormalDistribution(6.0, 1.0).logDensity(6.0);

        assertEquals(expected2, logProb2, 1e-5);
    }
}
