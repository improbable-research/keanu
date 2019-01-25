package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizerTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class AdamOptimizerTest {

    @Test
    public void canOptimizeSingleGaussianNetwork() {

        OptimizerTestCase testCase = new SingleGaussianTestCase(new long[0]);
        BayesianNetwork bayesianNetwork = testCase.getModel();
        KeanuProbabilisticWithGradientGraph model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);

        AdamOptimizer optimizer = AdamOptimizer.builder()
            .bayesianNetwork(model)
            .build();

        OptimizedResult result = optimizer.maxAPosteriori();

        testCase.assertMAP(result);
    }

    @Test
    public void canOptimizeSingleGaussianVectorNetwork() {

        SingleGaussianTestCase testCase = new SingleGaussianTestCase();

        BayesianNetwork bayesianNetwork = testCase.getModel();

        KeanuProbabilisticWithGradientGraph model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);

        AdamOptimizer optimizer = AdamOptimizer.builder()
            .alpha(0.1)
            .bayesianNetwork(model)
            .build();

        OptimizedResult result = optimizer.maxAPosteriori();

        testCase.assertMAP(result);
    }

    @Test
    public void canAddConvergenceChecker() {

        OptimizerTestCase testCase = new SingleGaussianTestCase(new long[0]);
        BayesianNetwork bayesianNetwork = testCase.getModel();
        KeanuProbabilisticWithGradientGraph model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);

        MutableInt i = new MutableInt(0);
        AdamOptimizer optimizer = AdamOptimizer.builder()
            .alpha(0.1)
            .bayesianNetwork(model)
            .convergenceChecker((gradient, theta, thetaNext) -> i.incrementAndGet() == 10)
            .build();

        optimizer.maxAPosteriori();
        
        assertTrue(i.getValue() == 10);
    }

    @Test
    public void canAddOnGradientHandler() {
        double mu = 10;
        GaussianVertex A = new GaussianVertex(mu, 0.1);
        A.setValue(0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());

        KeanuProbabilisticWithGradientGraph model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);

        AdamOptimizer optimizer = AdamOptimizer.builder()
            .bayesianNetwork(model)
            .build();

        MutableInt callCount = new MutableInt(0);
        optimizer.addGradientCalculationHandler((point, gradient) -> {
            callCount.increment();
        });

        optimizer.maxAPosteriori();

        assertTrue(callCount.getValue() > 0);
    }

}
