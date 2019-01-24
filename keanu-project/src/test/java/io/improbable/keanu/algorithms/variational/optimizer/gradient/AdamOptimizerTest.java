package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizerTest;
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

        AdamOptimizer optimizer = AdamOptimizer.builder()
            .alpha(0.001)
            .beta1(0.9)
            .beta2(0.999)
            .epsilon(1e-8)
            .bayesianNetwork(new BayesianNetwork(A.getConnectedGraph()))
            .build();

        optimizer.maxAPosteriori();

        assertEquals(mu, A.getValue().scalar(), 1e-2);
    }

    @Test
    public void calculateMaxLikelihood() {

        OptimizerTest.assertCanCalculateMaxLikelihood((bn) -> {

            AdamOptimizer optimizer = AdamOptimizer.builder()
                .alpha(0.001)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .bayesianNetwork(bn)
                .build();
            return optimizer;
        });

    }

    @Test
    public void canCalculateMaxAPosteriori() {

        OptimizerTest.assertCanCalculateMaxAPosteriori((bn) -> {

            AdamOptimizer optimizer = AdamOptimizer.builder()
                .alpha(0.001)
                .beta1(0.9)
                .beta2(0.999)
                .epsilon(1e-8)
                .bayesianNetwork(bn)
                .build();
            return optimizer;
        });

    }

}
