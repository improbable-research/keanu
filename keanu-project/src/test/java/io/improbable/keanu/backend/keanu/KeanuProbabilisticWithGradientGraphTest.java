package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

public class KeanuProbabilisticWithGradientGraphTest {

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
    public void canCalculateGradient() {
        A.setValue(0.5);
        B.setValue(0.25);
        BernoulliVertex C = new BernoulliVertex(A.times(B));
        C.observe(true);

        KeanuProbabilisticWithGradientGraph graph = KeanuGraphConverter.convertWithGradient(
            new BayesianNetwork(C.getConnectedGraph())
        );

        Map<String, DoubleTensor> gradients = graph.logProbGradients(null);


    }
}
