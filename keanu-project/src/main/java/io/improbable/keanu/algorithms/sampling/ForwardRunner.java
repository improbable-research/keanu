package io.improbable.keanu.algorithms.sampling;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.backend.ComputableGraph;

public class ForwardRunner {

    private final ProposalDistribution priorProposal;

    private ForwardRunner() {
        this.priorProposal = new PriorProposalDistribution();
    }

    /**
     * Samples from a Bayesian Network that only contains sampling information. No observations can have been made.
     * Samples are taken by calculating a linear ordering of the network and cascading the sampled values
     * through the network in priority order.
     *
     * @param graph the sampling bayesian network to sample from
     * @param fromVariables the vertices to sample from
     * @param sampleCount the number of samples to take
     * @return sampling samples of a bayesian network
     */
    public NetworkSamples sample(ComputableGraph graph,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount) {
        return sample(graph, fromVariables, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public NetworkSamples sample(ComputableGraph graph,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount,
                                        KeanuRandom random) {

        Set<Variable> chosenVariables = new HashSet<>(TopologicalSort.sort(fromVariables));
        Map<VariableReference, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            Proposal proposal = priorProposal.getProposal(chosenVariables, random);
            graph.compute(proposal.getProposalTo(), fromVariables.stream().map(Variable::getReference).collect(Collectors.toList()));
            takeSamples(samplesByVertex, fromVariables);
        }

        ArrayList<Double> logProb = new ArrayList<>(Collections.nCopies(sampleCount, 0.));
        return new NetworkSamples(samplesByVertex, logProb, sampleCount);
    }

    private static void takeSamples(Map<VariableReference, List> samples, List<? extends Variable> fromVertices) {
        fromVertices.forEach(variable -> addVariableValue(variable, samples));
    }

    private static void addVariableValue(Variable variable, Map<VariableReference, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<>());
        samplesForVertex.add(variable.getValue());
    }
}
