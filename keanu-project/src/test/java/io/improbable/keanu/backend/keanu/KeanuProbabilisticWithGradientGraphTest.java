package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class KeanuProbabilisticWithGradientGraphTest {

    UniformVertex A;
    UniformVertex B;

    @Before
    public void setup() {
        A = new UniformVertex(0.0, 1.0);
        A.setLabel("A");
        B = new UniformVertex(0.0, 1.0);
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

        DoubleTensor dLogProbWrtA = gradients.get("A");
        DoubleTensor dLogProbWrtB = gradients.get("B");

        //logProb = log(A*B)
        //dLogProb w.r.t A = B * 1/(A*B) = 1/A
        //dLogProb w.r.t B = A * 1/(A*B) = 1/B
        assertEquals(DoubleTensor.scalar(2.0), dLogProbWrtA);
        assertEquals(DoubleTensor.scalar(4.0), dLogProbWrtB);
    }
}
