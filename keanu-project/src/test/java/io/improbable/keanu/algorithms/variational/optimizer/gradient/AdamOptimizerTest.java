package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class AdamOptimizerTest {

    @Test
    public void canOptimizeSingleGaussianNetwork() {

        double mu = 10;
        GaussianVertex A = new GaussianVertex(mu, 0.1);
        A.setValue(0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());

        KeanuProbabilisticWithGradientGraph model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);

        AdamOptimizer optimizer = AdamOptimizer.builder()
            .bayesianNetwork(model)
            .build();

        optimizer.maxAPosteriori();

        assertEquals(mu, A.getValue().scalar(), 1e-2);
    }

}
