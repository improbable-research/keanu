package io.improbable.keanu.algorithms.sampling;

import com.google.common.collect.Sets;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.KeanuProposalApplicationStrategy;
import io.improbable.keanu.algorithms.mcmc.ProposalApplicationStrategy;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.backend.ComputableModel;
import io.improbable.keanu.backend.KeanuComputableModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import org.nd4j.base.Preconditions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class ForwardSampler {

    private static final double LOG_PROB_OF_PRIOR = 0.;
    private static final ProposalDistribution PRIOR_PROPOSAL = new PriorProposalDistribution();

    private ForwardSampler() {
    }

    /**
     * Samples from a Bayesian Network.
     * Samples are taken by sampling from the prior of the desired variables.
     * Probabilistic variables are sampled first.
     *
     * @param network the network to sample from
     * @param fromVariables the variables to sample from
     * @param sampleCount the number of samples to take
     * @return sampling samples of a computable graph
     */
    public static NetworkSamples sample(BayesianNetwork network,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount) {
        return sample(network, fromVariables, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples sample(BayesianNetwork network,
                                        List<? extends Variable> fromVariables,
                                        int sampleCount,
                                        KeanuRandom random) {

        Preconditions.checkArgument(network.getObservedVertices().size() == 0, "Cannot forward sample from a network with observations");

        List<Vertex> fromVertices = fromVariables.stream().map(v -> (Vertex) v).collect(Collectors.toList());

        Set<Vertex> probabilisticSubset = fromVertices.stream().filter(v -> v instanceof Probabilistic).collect(Collectors.toSet());
        Set<Vertex> nonProbabilisticSubset = fromVertices.stream().filter( v -> !(v instanceof Probabilistic)).collect(Collectors.toSet());

        upstreamOfProbabilisticDoesNotContainProbabilistic(probabilisticSubset);

        ComputableModel graph = new KeanuComputableModel(new HashSet<>(network.getAllVertices()));
        KeanuProposalApplicationStrategy proposalApplicationStrategy = new KeanuProposalApplicationStrategy(probabilisticSubset);

        return sample(graph, probabilisticSubset, nonProbabilisticSubset, proposalApplicationStrategy, sampleCount, random);
    }

    public static NetworkSamples sample(ComputableModel graph,
                                        Set<? extends Variable> probabilisticFromVariables,
                                        Set<? extends Variable> nonProbabilisticFromVariables,
                                        ProposalApplicationStrategy proposalApplicationStrategy,
                                        int sampleCount,
                                        KeanuRandom random) {

        Set<? extends Variable> allSampleFromVariables = Sets.union(probabilisticFromVariables, nonProbabilisticFromVariables);
        Map<VariableReference, List> samplesByVariable = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            Proposal probabilisticVariableProposal = PRIOR_PROPOSAL.getProposal(probabilisticFromVariables, random);
            proposalApplicationStrategy.apply(probabilisticVariableProposal);
            graph.compute(Collections.EMPTY_MAP, nonProbabilisticFromVariables.stream().map(Variable::getReference).collect(Collectors.toList()));
            takeSamples(samplesByVariable, allSampleFromVariables);
        }

        ArrayList<Double> logProb = new ArrayList<>(Collections.nCopies(sampleCount, LOG_PROB_OF_PRIOR));

        return new NetworkSamples(samplesByVariable, logProb, sampleCount);
    }

    private static void takeSamples(Map<VariableReference, List> samples, Set<? extends Variable> fromVariables) {
        fromVariables.forEach(variable -> addVariableValue(variable, samples));
    }

    private static void addVariableValue(Variable variable, Map<VariableReference, List> samples) {
        List samplesForVariable = samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<>());
        samplesForVariable.add(variable.getValue());
    }

    private static void upstreamOfProbabilisticDoesNotContainProbabilistic(Set<Vertex> vertices) {
        for (Vertex vertex : vertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, false);
            Set<Vertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();

            Preconditions.checkArgument(upstreamRandomVariables.size() == 1, "Vertex: [" + vertex + "] has a random variable in its upstream lambda section");
        }
    }
}
