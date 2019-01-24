package io.improbable.keanu.benchmarks;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticWithGradientGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.AdamOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.util.ProgressBar;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.openjdk.jmh.annotations.*;

@State(Scope.Benchmark)
public class OptimizerBenchmark {

    public enum OptimizerType {
        ADAM, APACHE_GRADIENT, APACHE_NON_GRADIENT
    }

    @Param({"ADAM", "APACHE_GRADIENT", "APACHE_NON_GRADIENT"})
    public OptimizerType optimizerType;

    private Optimizer optimizer;

    @Setup
    public void setup() {

        ProgressBar.disable();

        GaussianVertex A = new GaussianVertex(10, 0.1);
        A.setValue(0.0);
        GaussianVertex B = new GaussianVertex(5, 1);
        B.setValue(0.0);
        GaussianVertex C = new GaussianVertex(A.times(B), 0.1);
        C.observe(30.0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());

        switch (optimizerType) {
            case ADAM:
                optimizer = AdamOptimizer.builder()
                    .alpha(0.001)
                    .beta1(0.9)
                    .beta2(0.999)
                    .epsilon(1e-8)
                    .bayesianNetwork(bayesianNetwork)
                    .build();
                break;

            case APACHE_GRADIENT:
                optimizer = GradientOptimizer.builder()
                    .bayesianNetwork(new KeanuProbabilisticWithGradientGraph(bayesianNetwork))
                    .build();
                break;
            case APACHE_NON_GRADIENT:
                optimizer = NonGradientOptimizer.builder()
                    .bayesianNetwork(new KeanuProbabilisticWithGradientGraph(bayesianNetwork))
                    .build();
                break;
        }
    }

    @Benchmark
    public double baseline() {
        return optimizer.maxAPosteriori();
    }

}
