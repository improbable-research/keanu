package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfGaussianVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertTrue;

public class NonGradientOptimizerTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void doesCallOnFitnessHandler() {
        AtomicInteger timesCalled = new AtomicInteger(0);
        NonGradientOptimizer optimizer = Keanu.Optimizer.NonGradient.ofConnectedGraph(new GaussianVertex(new GaussianVertex(0, 1), 1));
        optimizer.addFitnessCalculationHandler((point, fitness) -> timesCalled.incrementAndGet());
        optimizer.maxAPosteriori();
        assertTrue(timesCalled.get() > 0);
    }

    @Test
    public void doesCheckForZeroProbability() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Cannot start optimizer on zero probability network");
        runMAPOnZeroProbability(true);
    }

    @Test
    public void doesNotCheckForZeroProbabilityWhenChecksAreTurnedOff() {
        runMAPOnZeroProbability(false);
    }

    private void runMAPOnZeroProbability(boolean enableCheck) {
        HalfGaussianVertex B = new HalfGaussianVertex(1);
        HalfGaussianVertex A = new HalfGaussianVertex(B);
        A.observe(-1);

        ProbabilisticModel model = new KeanuProbabilisticModel(A.getConnectedGraph());

        NonGradientOptimizer optimizer = Keanu.Optimizer.NonGradient.builderFor(A.getConnectedGraph())
            .probabilisticModel(model)
            .algorithm((vars, fitness) -> new OptimizedResult(null, 0))
            .checkInitialFitnessConditions(enableCheck)
            .build();

        optimizer.maxAPosteriori();
    }

}
