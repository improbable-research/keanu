package io.improbable.keanu.backend;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@UtilityClass
public class ProbabilisticGraphConverter {


    public static Optional<VariableReference> convertLogProbObservation(BayesianNetwork network,
                                                                        ComputableGraphBuilder<?> graphBuilder) {
        return addLogProbCalculation(
            graphBuilder,
            network.getObservedVertices()
        );
    }

    public static VariableReference convertLogProbPrior(BayesianNetwork network,
                                                        ComputableGraphBuilder<?> graphBuilder) {
        return addLogProbCalculation(
            graphBuilder,
            network.getLatentVertices()
        ).orElseThrow(() -> new IllegalArgumentException("Network must contain latent variables"));
    }

//    static void convert(BayesianNetwork network, ComputableGraphBuilder<?> graphBuilder) {
//
//        graphBuilder.convert(network.getVertices());
//
//        VariableReference priorLogProbReference = addLogProbCalculation(
//            graphBuilder,
//            network.getLatentVertices()
//        ).orElseThrow(() -> new IllegalArgumentException("Network must contain latent variables"));
//
//        Optional<VariableReference> logLikelihoodReference = addLogProbCalculation(
//            graphBuilder,
//            network.getObservedVertices()
//        );
//
//        VariableReference logProbReference = logLikelihoodReference
//            .map(ll -> graphBuilder.add(ll, priorLogProbReference))
//            .orElse(priorLogProbReference);
//
//    }

    private static Optional<VariableReference> addLogProbCalculation(ComputableGraphBuilder<?> graphBuilder,
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

        Optional<VariableReference> logProbSummation = logProbOps.stream()
            .reduce(graphBuilder::add);

        return logProbSummation;
    }

    /**
     * @param logProbGraph the graph to add that represents the logProb calculation
     * @param graphBuilder the builder that contains state for the probabilistic graph building so far.
     */
    private static VariableReference addLogProbGraph(LogProbGraph logProbGraph,
                                                     ComputableGraphBuilder<?> graphBuilder) {

        graphBuilder.connect(logProbGraph.getInputs());
        graphBuilder.convert(logProbGraph.getLogProbOutput().getConnectedGraph());
        return logProbGraph.getLogProbOutput().getReference();
    }

}
