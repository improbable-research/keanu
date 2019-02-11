package io.improbable.keanu.algorithms.sampling;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.NonProbabilistic;
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

public class ForwardSampler {

    private static final double LOG_PROB_OF_PRIOR = 0.;
    private static final ProposalDistribution PRIOR_PROPOSAL = new PriorProposalDistribution();

    public ForwardSampler() {
    }

    /**
     * Samples from a Bayesian Network.
     * Samples are taken by sampling from the prior of the desired variables.
     * Probabilistic variables are sampled first.
     * Non probabilistic variables are computed second.
     *
     * @param network the network to sample from
     * @param fromVertices the variables to sample from
     * @param sampleCount the number of samples to take
     * @return sampling samples of a computable graph
     */

    public NetworkSamples sample(BayesianNetwork network,
                                 List<Vertex> fromVertices,
                                 int sampleCount) {
        return sample(network, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public NetworkSamples sample(BayesianNetwork network,
                                        List<Vertex> fromVertices,
                                        int sampleCount,
                                        KeanuRandom random) {

        Map<VariableReference, List<?>> samplesByVertex = new HashMap<>();
        List<Vertex> observedVertices = network.getObservedVertices();
        upstreamOfObservedDoesNotContainProbabilistic(observedVertices);

        Set<Vertex> allUpstreamVertices = upstreamVertices(fromVertices);
        List<Vertex> sortedVertices = TopologicalSort.sort(allUpstreamVertices);

        sortedVertices = pruneNonProbabilistic(sortedVertices);

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            for (Vertex vertex : sortedVertices) {
                if (vertex instanceof Probabilistic) {
                    vertex.setValue(((Probabilistic) vertex).sample(random));
                } else if (vertex instanceof NonProbabilistic){
                    vertex.setValue(((NonProbabilistic) vertex).calculate());
                }
            }
            takeSamples(samplesByVertex, fromVertices);
        }

        ArrayList<Double> logProb = new ArrayList<>(Collections.nCopies(sampleCount, LOG_PROB_OF_PRIOR));
        return new NetworkSamples(samplesByVertex, logProb, sampleCount);
    }

    private List<Vertex> pruneNonProbabilistic(List<Vertex> vertices) {
        for (int i = 0; i < vertices.size(); i++) {
            Vertex vertex = vertices.get(i);
            if (!(vertex instanceof Probabilistic)) {
                vertices.remove(i);
            } else {
                break;
            }
        }
        return vertices;
    }

    private Set<Vertex> upstreamVertices(List<Vertex> fromVertices) {
        Set<Vertex> upstream = new HashSet<>();
        for (Vertex vertex : fromVertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, true);
            Set<Vertex> vertexUpsteam = upstreamLambdaSection.getAllVertices();
            upstream = Sets.union(upstream, vertexUpsteam);
        }
        return upstream;
    }

    private void takeSamples(Map<VariableReference, List<?>> samples, List<Vertex> fromVertices) {
        fromVertices.forEach(variable -> addVariableValue(variable, samples));
    }

    private void addVariableValue(Variable variable, Map<VariableReference, List<?>> samples) {
        List samplesForVariable = samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<>());
        samplesForVariable.add(variable.getValue());
    }

    private void upstreamOfObservedDoesNotContainProbabilistic(List<Vertex> vertices) {
        for (Vertex vertex : vertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, false);
            Set<Vertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();

            Preconditions.checkArgument(upstreamRandomVariables.size() == 1, "Vertex: [" + vertex + "] has a random variable in its upstream lambda section");
        }
    }
}
