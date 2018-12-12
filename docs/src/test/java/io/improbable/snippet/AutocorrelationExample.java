package io.improbable.snippet;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class AutocorrelationExample {
    public static void main(String[] args) {
        scalarAutocorrelationExample();
        tensorAutocorrelationExample();
    }

    private static void scalarAutocorrelationExample() {
DoubleVertex A = new GaussianVertex(20.0, 1.0);
DoubleVertex B = new GaussianVertex(20.0, 1.0);
DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
C.observe(43.0);
A.setValue(20.0);
B.setValue(20.0);
BayesianNetwork bayesNet = new BayesianNetwork(C.getConnectedGraph());

//%%SNIPPET_START%% ScalarAutocorrelation
NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
    bayesNet,
    bayesNet.getLatentVertices(),
    100
);
DoubleTensor autocorrelation = posteriorSamples.getDoubleTensorSamples(A).getAutocorrelation();
//%%SNIPPET_END%% ScalarAutocorrelation
    }

    private static void tensorAutocorrelationExample() {
DoubleVertex A = new GaussianVertex(new long[]{1, 5}, 20.0, 1.0);
DoubleVertex B = new GaussianVertex(new long[]{1, 5}, 20.0, 1.0);
DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
BayesianNetwork bayesNet = new BayesianNetwork(C.getConnectedGraph());
C.observe(new double[]{1, 4, 5, 7, 8});
bayesNet.probeForNonZeroProbability(100);

//%%SNIPPET_START%% TensorAutocorrelation
NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
    bayesNet,
    bayesNet.getLatentVertices(),
    100
);
DoubleTensor autocorrelation = posteriorSamples.getDoubleTensorSamples(A).getAutocorrelation(0,1);
//%%SNIPPET_END%% TensorAutocorrelation
    }
}
