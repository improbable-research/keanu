package io.improbable.keanu.benchmarks;

import io.improbable.keanu.algorithms.variational.optimizer.ConvergenceChecker;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ConjugateGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.BOBYQA;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;

@State(Scope.Benchmark)
public class OptimizerBenchmark {

    public enum OptimizerType {
        ADAM_OPTIMIZER, CONJUGATE_GRADIENT, BOBYQA_OPTIMIZER
    }

    @Param({"ADAM_OPTIMIZER", "CONJUGATE_GRADIENT", "BOBYQA_OPTIMIZER"})
    public OptimizerType optimizerType;

    private Optimizer optimizer;

    @Setup
    public void setup() {

        StatusBar.disable();

        GaussianVertex A = new GaussianVertex(10, 0.1);
        A.setValue(0.0);
        GaussianVertex B = new GaussianVertex(5, 1);
        B.setValue(0.0);
        GaussianVertex C = new GaussianVertex(A.times(B), 0.1);
        C.observe(30.0);

        KeanuProbabilisticModelWithGradient gradientGraph = new KeanuProbabilisticModelWithGradient(A.getConnectedGraph());

        switch (optimizerType) {
            case ADAM_OPTIMIZER:
                optimizer = GradientOptimizer.builder()
                    .probabilisticModel(gradientGraph)
                    .algorithm(Adam.builder()
                        .convergenceChecker(ConvergenceChecker.relativeChecker(ConvergenceChecker.Norm.L2, 1e-5))
                        .build())
                    .build();
                break;

            case CONJUGATE_GRADIENT:
                optimizer = GradientOptimizer.builder()
                    .probabilisticModel(gradientGraph)
                    .algorithm(ConjugateGradient.builder().build())
                    .build();
                break;
            case BOBYQA_OPTIMIZER:
                optimizer = NonGradientOptimizer.builder()
                    .probabilisticModel(gradientGraph)
                    .algorithm(BOBYQA.builder().build())
                    .build();
                break;
        }
    }

    private OptimizedResult result;

    @Benchmark
    public OptimizedResult findMap() {
        result = optimizer.maxAPosteriori();
        return result;
    }

    @TearDown
    public void checkResult() {
        assertEquals(result.getFitness(), -0.1496, 1e-2);
    }

    private void assertEquals(double optimizedFitness, double expected, double eps) {

        if (Math.abs(optimizedFitness - expected) >= eps) {
            throw new IllegalStateException(optimizedFitness + " not " + expected);
        }
    }

}
