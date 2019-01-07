package io.improbable.keanu.backend;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.LogProbAsAGraphable;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@UtilityClass
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
                addLogProbFrom(logProbGraph, graphBuilder);
                logProbOps.add(logProbGraph.getLogProbOutput().getReference());

            } else {
                throw new IllegalArgumentException("Vertex type " + visiting.getClass() + " logProb as a graph not supported");
            }
        }

        return addLogProbSumTotal(logProbOps, graphBuilder);
    }

    /**
     * @param logProbGraph the graph to add that represents the logProb calculation
     * @param graphBuilder the builder that contains state for the probabilistic graph building so far.
     */
    private static void addLogProbFrom(LogProbGraph logProbGraph,
                                       ProbabilisticGraphBuilder graphBuilder) {

        Map<Vertex<?>, Vertex<?>> inputs = logProbGraph.getInputs();
        graphBuilder.connect(inputs);
        graphBuilder.convert(((Vertex) logProbGraph.getLogProbOutput()).getConnectedGraph());
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
