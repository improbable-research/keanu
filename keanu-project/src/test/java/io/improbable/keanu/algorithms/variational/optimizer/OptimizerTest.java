package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class OptimizerTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

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

        BiConsumer<Map<VariableReference, DoubleTensor>, Double> fitnessHandler = mock(BiConsumer.class);

        optimizer.addFitnessCalculationHandler(fitnessHandler);
        optimizer.removeFitnessCalculationHandler(fitnessHandler);

        optimizer.maxAPosteriori();

        assertFalse(didCallFitness.get());
        verifyNoMoreInteractions(fitnessHandler);
    }
}
