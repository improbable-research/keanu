package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class OptimizerTest {

    @Test
    public void gradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getGradientOptimizer());
    }

    @Test
    public void nonGradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getNonGradientOptimizer());
    }

    @Test
    public void gradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getGradientOptimizer());
    }

    @Test
    public void nonGradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getNonGradientOptimizer());
    }

    private Function<BayesianNetwork, Optimizer> getGradientOptimizer() {
        return (bayesNet) -> GradientOptimizer.builder()
            .bayesianNetwork(bayesNet)
            .build();
    }

    private Function<BayesianNetwork, Optimizer> getNonGradientOptimizer() {
        return (bayesNet) -> NonGradientOptimizer.builder()
            .bayesianNetwork(bayesNet)
            .build();
    }

    private void assertCanCalculateMaxLikelihood(Function<BayesianNetwork, Optimizer> optimizerMapper) {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(44.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        Optimizer optimizer = optimizerMapper.apply(bayesNet);

        optimizer.maxLikelihood();
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(44, maxA + maxB, 0.1);
    }

    public void assertCanCalculateMaxAPosteriori(Function<BayesianNetwork, Optimizer> optimizerMapper) {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(21.5);
        B.setAndCascade(21.5);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        Optimizer optimizer = optimizerMapper.apply(bayesNet);

        optimizer.maxAPosteriori();
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }

    @Test
    public void gradientOptimizerCanRemoveFitnessCalculationHandler() {
        GaussianVertex gaussianVertex = new GaussianVertex(0, 1);
        GradientOptimizer optimizer = GradientOptimizer.of(gaussianVertex.getConnectedGraph());
        canRemoveFitnessCalculationHandler(optimizer);
    }

    @Test
    public void nonGradientOptimizerCanRemoveFitnessCalculationHandler() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        A.plus(B);
        NonGradientOptimizer optimizer = NonGradientOptimizer.of(A.getConnectedGraph());
        canRemoveFitnessCalculationHandler(optimizer);
    }

    private void canRemoveFitnessCalculationHandler(Optimizer optimizer) {

        AtomicBoolean didCallFitness = new AtomicBoolean(false);

        BiConsumer<double[], Double> fitnessHandler = mock(BiConsumer.class);

        optimizer.addFitnessCalculationHandler(fitnessHandler);
        optimizer.removeFitnessCalculationHandler(fitnessHandler);

        optimizer.maxAPosteriori();

        assertFalse(didCallFitness.get());
        verifyNoMoreInteractions(fitnessHandler);
    }
}
