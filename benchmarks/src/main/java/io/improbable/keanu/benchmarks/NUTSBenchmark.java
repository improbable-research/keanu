package io.improbable.keanu.benchmarks;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class NUTSBenchmark {

    private ProbabilisticModelWithGradient model;

    @Setup
    public void setup() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        GaussianVertex D = new GaussianVertex((A.multiply(A)).plus(B.multiply(B)), 0.03);
        D.observe(0.5);

        model = new KeanuProbabilisticModelWithGradient(new BayesianNetwork(A.getConnectedGraph()));
        StatusBar.disable();
    }

    @Benchmark
    public NetworkSamples takeSamples() {

        final int samples = 1000;
        NUTS nuts = NUTS.builder()
            .adaptCount(samples)
            .build();

        return nuts.getPosteriorSamples(model, samples);
    }
}
