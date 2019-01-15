package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.network.BayesianNetwork;

import java.util.Collections;
import java.util.List;

public interface PosteriorSamplingAlgorithm {

    default NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesianNetwork,
                                               Variable vertexToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesianNetwork, int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, bayesianNetwork.getLatentVariables(), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                               Variable vertexToSampleFrom,
                                               int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, Collections.singletonList(vertexToSampleFrom), sampleCount);
    }

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                       List<? extends Variable> verticesToSampleFrom,
                                       int sampleCount) {
        return getPosteriorSamples(new KeanuProbabilisticGraph(bayesNet), verticesToSampleFrom, sampleCount);
    }

    default NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork, int sampleCount) {
        KeanuProbabilisticGraph graph = new KeanuProbabilisticGraph(bayesianNetwork);
        return getPosteriorSamples(graph, graph.getLatentVariables(), sampleCount);
    }

    NetworkSamples getPosteriorSamples(ProbabilisticGraph bayesNet,
                                       List<? extends Variable> verticesToSampleFrom,
                                       int sampleCount);

}
