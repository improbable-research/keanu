package io.improbable.keanu.backend.tensorflow;

import org.junit.Test;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorflowGraphConverterTest {

    @Test
    public void canRunSimpleAddition() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(2);
        DoubleVertex B = new GaussianVertex(1, 1);
        B.setValue(3);

        DoubleVertex C = A.plus(B);

        TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
    }
}
