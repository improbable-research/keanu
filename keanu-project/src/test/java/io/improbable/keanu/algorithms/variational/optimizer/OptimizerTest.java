package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.backend.tensorflow.TensorflowProbabilisticGraph;
import io.improbable.keanu.backend.tensorflow.TensorflowProbabilisticGraphWithGradientFactory;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class OptimizerTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void keanuGradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getKeanuGradientOptimizer());
    }

    @Test
    public void tensorflowGradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getTensorflowGradientOptimizer());
    }

    @Test
    public void keanuNonGradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getKeanuNonGradientOptimizer());
    }

    @Test
    public void tensorflowNonGradientOptimizerCanMLE() {
        assertCanCalculateMaxLikelihood(getTensorflowNonGradientOptimizer());
    }

    @Test
    public void keanuGradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getKeanuGradientOptimizer());
    }

    @Test
    public void tensorflowGradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getTensorflowGradientOptimizer());
    }

    @Test
    public void keanuNonGradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getKeanuNonGradientOptimizer());
    }

    @Test
    public void tensorflowNonGradientOptimizerCanMAP() {
        assertCanCalculateMaxAPosteriori(getTensorflowNonGradientOptimizer());
    }

    private Function<BayesianNetwork, Optimizer> getKeanuGradientOptimizer() {
        return (bayesNet) -> KeanuOptimizer.Gradient.of(bayesNet);
    }

    private Function<BayesianNetwork, Optimizer> getKeanuNonGradientOptimizer() {
        return (bayesNet) -> KeanuOptimizer.NonGradient.of(bayesNet);
    }

    private Function<BayesianNetwork, Optimizer> getTensorflowGradientOptimizer() {
        return (bayesNet) -> GradientOptimizer.builder()
            .bayesianNetwork(TensorflowProbabilisticGraphWithGradientFactory.convert(bayesNet))
            .build();
    }

    private Function<BayesianNetwork, Optimizer> getTensorflowNonGradientOptimizer() {
        return (bayesNet) -> NonGradientOptimizer.builder()
            .bayesianNetwork(TensorflowProbabilisticGraph.convert(bayesNet))
            .build();
    }

    private void assertCanCalculateMaxLikelihood(Function<BayesianNetwork, Optimizer> optimizerMapper) {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(44.0);

        BayesianNetwork bayesNet = new BayesianNetwork(A.getConnectedGraph());

        Optimizer optimizer = optimizerMapper.apply(bayesNet);

        OptimizedResult optimizedResult = optimizer.maxLikelihood();
        double maxA = optimizedResult.get(A.getReference()).scalar();
        double maxB = optimizedResult.get(B.getReference()).scalar();

        assertEquals(44, maxA + maxB, 0.1);
    }

    public void assertCanCalculateMaxAPosteriori(Function<BayesianNetwork, Optimizer> optimizerMapper) {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(21.5);
        B.setAndCascade(21.5);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(A.getConnectedGraph());

        Optimizer optimizer = optimizerMapper.apply(bayesNet);

        OptimizedResult optimizedResult = optimizer.maxAPosteriori();
        double maxA = optimizedResult.get(A.getReference()).scalar();
        double maxB = optimizedResult.get(B.getReference()).scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }

    @Test
    public void gradientOptimizerCanRemoveFitnessCalculationHandler() {
        GaussianVertex gaussianVertex = new GaussianVertex(0, 1);
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.of(gaussianVertex.getConnectedGraph());
        canRemoveFitnessCalculationHandler(optimizer);
    }

    @Test
    public void nonGradientOptimizerCanRemoveFitnessCalculationHandler() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);
        A.plus(B);
        NonGradientOptimizer optimizer = KeanuOptimizer.NonGradient.of(A.getConnectedGraph());
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
