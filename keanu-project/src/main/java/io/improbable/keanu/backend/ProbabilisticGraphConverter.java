package io.improbable.keanu.backend;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ProbabilisticGraphConverter {

    public static <T extends ProbabilisticGraph> T convert(BayesianNetwork network, ProbabilisticGraphBuilder<T> graphBuilder) {

        graphBuilder.convert(network.getVertices().get(0).getConnectedGraph());

        VariableReference priorLogProbReference = addLogProbCalculation(
            graphBuilder,
            network.getLatentVertices()
        );

        VariableReference logProbReference;
        VariableReference logLikelihoodReference = null;

        if (!network.getObservedVertices().isEmpty()) {
            logLikelihoodReference = addLogProbCalculation(
                graphBuilder,
                network.getObservedVertices()
            );

            logProbReference = graphBuilder.add(logLikelihoodReference, priorLogProbReference);
        } else {
            logProbReference = priorLogProbReference;
        }

        graphBuilder.logProb(logProbReference);
        graphBuilder.logLikelihood(logLikelihoodReference);

        return graphBuilder.build();
    }

    private static <T extends ProbabilisticGraph> VariableReference addLogProbCalculation(ProbabilisticGraphBuilder<T> graphBuilder,
                                                                                          List<Vertex> probabilisticVertices) {
        List<VariableReference> logProbOps = new ArrayList<>();

        for (Vertex visiting : probabilisticVertices) {

            if (visiting instanceof LogProbAsAGraphable) {
                LogProbGraph logProbGraph = ((LogProbAsAGraphable) visiting).logProbGraph();
                VariableReference logProbFromVisiting = addLogProbFrom(logProbGraph, graphBuilder);
                logProbOps.add(logProbFromVisiting);

            } else {
                throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
            }
        }

        return addLogProbSumTotal(logProbOps, graphBuilder);
    }

    private static <T extends ProbabilisticGraph> VariableReference addLogProbFrom(LogProbGraph logProbGraph,
                                                                                   ProbabilisticGraphBuilder<T> graphBuilder) {

        Map<Vertex<?>, Vertex<?>> inputs = logProbGraph.getInputs();

        //setup graph connection
        for (Map.Entry<Vertex<?>, Vertex<?>> input : inputs.entrySet()) {
            graphBuilder.alias(input.getValue().getReference(), input.getKey().getReference());
        }

        List<Vertex> topoSortedVertices = (logProbGraph.getLogProbOutput().getConnectedGraph()).stream()
            .sorted(Comparator.comparing(Vertex::getId))
            .collect(Collectors.toList());

        HashSet<Vertex<?>> logProbInputs = new HashSet<>(inputs.values());

        for (Vertex visiting : topoSortedVertices) {

            if (!logProbInputs.contains(visiting)) {
                graphBuilder.convert(visiting);
            }
        }

        return logProbGraph.getLogProbOutput().getReference();
    }

    private static <T extends ProbabilisticGraph> VariableReference addLogProbSumTotal(List<VariableReference> logProbOps,
                                                                                       ProbabilisticGraphBuilder<T> graphBuilder) {

        VariableReference totalLogProb = logProbOps.get(0);

        if (logProbOps.size() == 1) {
            return totalLogProb;
        }

        for (int i = 1; i < logProbOps.size() - 1; i++) {
            VariableReference logProbContrib = logProbOps.get(i);
            totalLogProb = graphBuilder.add(totalLogProb, logProbContrib);
        }

        VariableReference lastLogProbContrib = logProbOps.get(logProbOps.size() - 1);
        return graphBuilder.add(totalLogProb, lastLogProbContrib);
    }

}
