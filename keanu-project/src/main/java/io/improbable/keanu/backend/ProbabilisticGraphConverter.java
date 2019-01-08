package io.improbable.keanu.backend;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@UtilityClass
public class ProbabilisticGraphConverter {

    public static <T extends ProbabilisticGraph> T convert(BayesianNetwork network, ProbabilisticGraphBuilder<T> graphBuilder) {

        graphBuilder.convert(network.getVertices().get(0).getConnectedGraph());

        VariableReference priorLogProbReference = addLogProbCalculation(
            graphBuilder,
            network.getLatentVertices()
        ).orElseThrow(() -> new IllegalArgumentException("Network must contain latent variables"));

        Optional<VariableReference> logLikelihoodReference = addLogProbCalculation(
            graphBuilder,
            network.getObservedVertices()
        );

        VariableReference logProbReference = logLikelihoodReference
            .map(ll -> graphBuilder.add(ll, priorLogProbReference))
            .orElse(priorLogProbReference);

        graphBuilder.logProb(logProbReference);
        logLikelihoodReference.ifPresent(graphBuilder::logLikelihood);

        return graphBuilder.build();
    }

    private static <T extends ProbabilisticGraph> Optional<VariableReference> addLogProbCalculation(ProbabilisticGraphBuilder<T> graphBuilder,
                                                                                                    List<Vertex> probabilisticVertices) {
        List<VariableReference> logProbOps = probabilisticVertices.stream()
            .map(visiting -> {
                if (visiting instanceof LogProbAsAGraphable) {
                    LogProbGraph logProbGraph = ((LogProbAsAGraphable) visiting).logProbGraph();
                    return addLogProbGraph(logProbGraph, graphBuilder);
                } else {
                    throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
                }
            }).collect(Collectors.toList());

        return addLogProbSumTotal(logProbOps, graphBuilder);
    }

    /**
     * @param logProbGraph the graph to add that represents the logProb calculation
     * @param graphBuilder the builder that contains state for the probabilistic graph building so far.
     */
    private static VariableReference addLogProbGraph(LogProbGraph logProbGraph,
                                                     ProbabilisticGraphBuilder graphBuilder) {

        Map<Vertex<?>, Vertex<?>> inputs = logProbGraph.getInputs();
        graphBuilder.connect(inputs);
        graphBuilder.convert(((Vertex) logProbGraph.getLogProbOutput()).getConnectedGraph());
        return logProbGraph.getLogProbOutput().getReference();
    }

    private static <T extends ProbabilisticGraph> Optional<VariableReference> addLogProbSumTotal(List<VariableReference> logProbOps,
                                                                                                 ProbabilisticGraphBuilder<T> graphBuilder) {
        return logProbOps.stream()
            .reduce((a, b) -> {
                if (a == null) {
                    return b;
                } else {
                    return graphBuilder.add(a, b);
                }
            });
    }


}
