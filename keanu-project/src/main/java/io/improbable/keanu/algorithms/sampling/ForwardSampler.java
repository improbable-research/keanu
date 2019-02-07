package io.improbable.keanu.algorithms.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.nd4j.base.Preconditions;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import kotlin.jvm.internal.Lambda;

public class ForwardSampler {

    private static final double LOG_PROB_OF_PRIOR = 0.;
    private static final ProposalDistribution PRIOR_PROPOSAL = new PriorProposalDistribution();

    private ForwardSampler() {
    }

    /**
     * Samples from a Computable Graph.
     * Samples are taken by sampling from the prior of the provided random variables and propagating these values
     * to any non probabilistic variables that the user may also want to sample from
     *
     * @param graph the computable graph to sample from
     * @param fromVariables the variables to sample from, can be probabilistic or non probabilistic
     * @param sampleCount the number of samples to take
     * @return sampling samples of a computable graph
     */
    public static NetworkSamples sample(ComputableGraph graph,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount) {
        return sample(graph, fromVariables, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples sample(ComputableGraph graph,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount,
                                        KeanuRandom random) {

        Set<Variable> probabilisticSubset = fromVariables.stream().filter(v -> v instanceof Probabilistic).collect(Collectors.toSet());
        upstreamDoesNotContainRandomVariables(probabilisticSubset);
        Map<VariableReference, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            Proposal proposal = PRIOR_PROPOSAL.getProposal(probabilisticSubset, random);
            graph.compute(proposal.getProposalTo(), fromVariables.stream().map(Variable::getReference).collect(Collectors.toList()));
            takeSamples(samplesByVertex, fromVariables);
        }

        ArrayList<Double> logProb = new ArrayList<>(Collections.nCopies(sampleCount, LOG_PROB_OF_PRIOR));
        return new NetworkSamples(samplesByVertex, logProb, sampleCount);
    }

    private static void upstreamDoesNotContainRandomVariables(Set<Variable> vertices) {
        for (Variable variable : vertices) {
            Vertex vertex = (Vertex) variable;
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, false);
            Set<Vertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();

            Preconditions.checkArgument(upstreamRandomVariables.size() == 1, "Vertex: [" + vertex + "] has a random variable in its upstream lambda section");
        }
    }

    private static void takeSamples(Map<VariableReference, List> samples, List<? extends Variable> fromVertices) {
        fromVertices.forEach(variable -> addVariableValue(variable, samples));
    }

    private static void addVariableValue(Variable variable, Map<VariableReference, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<>());
        samplesForVertex.add(variable.getValue());
    }
}
