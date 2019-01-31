package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GradientOptimizerTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void doesCallOnFitnessAndOnGradientHandler() {
        AtomicInteger fitnessTimesCalled = new AtomicInteger(0);
        AtomicInteger gradientTimesCalled = new AtomicInteger(0);
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.ofConnectedGraph(
            new GaussianVertex(0, 1)
        );
        optimizer.addFitnessCalculationHandler((point, fitness) -> fitnessTimesCalled.incrementAndGet());
        optimizer.addGradientCalculationHandler((point, fitness) -> gradientTimesCalled.incrementAndGet());
        optimizer.maxAPosteriori();

        assertTrue(fitnessTimesCalled.get() > 0);
        assertTrue(gradientTimesCalled.get() > 0);
    }

    @Test(expected = UnsupportedOperationException.class)
    public void errorOnDiscreteLatents() {
        PoissonVertex v1 = new PoissonVertex(15);
        PoissonVertex v2 = new PoissonVertex(v1);

        GradientOptimizer optimizer = KeanuOptimizer.Gradient.ofConnectedGraph(v1);
    }

    @Test
    public void shouldAllowObservationChange() {

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(A.abs(), 1);
        A.observe(2.0);

        GradientOptimizer optimizer = KeanuOptimizer.Gradient.ofConnectedGraph(B);
        optimizer.maxAPosteriori();

        assertEquals(2.0, B.getValue().scalar(), 1e-5);

        A.observe(3.0);

        GradientOptimizer optimizerAfterObserve = KeanuOptimizer.Gradient.ofConnectedGraph(B);
        optimizerAfterObserve.maxAPosteriori();

        assertEquals(3.0, B.getValue().scalar(), 1e-5);

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

    @Test
    public void doesCheckForFlatGradient() {
        expectedException.expect(IllegalStateException.class);
        expectedException.expectMessage("The initial gradient is very flat. The largest gradient is 0.0");
        runMAPOnZeroGradient(true);
    }

    @Test
    public void doesNotCheckForFlatGradientWhenChecksAreTurnedOff() {
        runMAPOnZeroGradient(false);
    }

    private void runMAPOnZeroProbability(boolean enableCheck) {
        HalfGaussianVertex B = new HalfGaussianVertex(1);
        HalfGaussianVertex A = new HalfGaussianVertex(B);
        A.observe(-1);

        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(A.getConnectedGraph());
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.builderFor(A.getConnectedGraph())
            .probabilisticModel(model)
            .algorithm((vars, fitness, gradient) -> new OptimizedResult(null, 0))
            .checkInitialFitnessConditions(enableCheck)
            .build();

        optimizer.maxAPosteriori();
    }

    private void runMAPOnZeroGradient(boolean enableCheck) {
        UniformVertex A = new UniformVertex(0, 1);
        A.setValue(0.5);

        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(A.getConnectedGraph());
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.builderFor(A.getConnectedGraph())
            .probabilisticModel(model)
            .algorithm((vars, fitness, gradient) -> new OptimizedResult(null, 0))
            .checkInitialFitnessConditions(enableCheck)
            .build();

        optimizer.maxAPosteriori();
    }
}
