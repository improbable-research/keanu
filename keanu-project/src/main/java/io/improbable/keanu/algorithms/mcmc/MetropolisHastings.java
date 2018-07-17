package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.proposals.PriorProposal;
import io.improbable.keanu.algorithms.mcmc.proposals.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposals.ProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.network.NetworkSnapshot;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Metropolis Hastings is a Markov Chain Monte Carlo method for obtaining samples from a probability distribution
 */
@Builder
public class MetropolisHastings implements PosteriorSamplingAlgorithm {

    public static final double LOG_ZERO_PROBABILITY = Double.NEGATIVE_INFINITY;

    public static MetropolisHastings withDefaultConfig() {
        return MetropolisHastings.builder().build();
    }

    @Builder.Default
    private final ProposalDistribution proposalDistribution = PriorProposal.SINGLETON;

    @Override
    public NetworkSamples getPosteriorSamples(BayesianNetwork bayesianNetwork,
                                              List<? extends Vertex> verticesToSampleFrom,
                                              int sampleCount) {
        return getPosteriorSamples(bayesianNetwork, verticesToSampleFrom, sampleCount, KeanuRandom.getDefaultRandom());
    }

    /**
     * @param bayesianNetwork      a bayesian network containing latent vertices
     * @param verticesToSampleFrom the vertices to include in the returned samples
     * @param sampleCount          number of samples to take using the algorithm
     * @param random               the source of randomness
     * @return Samples for each vertex ordered by MCMC iteration
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesianNetwork,
                                              final List<? extends Vertex> verticesToSampleFrom,
                                              final int sampleCount,
                                              final KeanuRandom random) {
        checkBayesNetInHealthyState(bayesianNetwork);

        Map<Long, List<?>> samplesByVertex = new HashMap<>();
        List<Vertex> latentVertices = bayesianNetwork.getLatentVertices();
        Map<Vertex, LambdaSection> affectedVerticesCache = getVerticesAffectedByLatents(latentVertices, true);

        double totalLogProbability = bayesianNetwork.getLogOfMasterP();
        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {

            Vertex<?> chosenVertex = latentVertices.get(sampleNum % latentVertices.size());
            totalLogProbability = nextSample(
                Collections.singleton(chosenVertex),
                affectedVerticesCache,
                proposalDistribution,
                totalLogProbability,
                1.0,
                random
            );

            takeSamples(samplesByVertex, verticesToSampleFrom);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    /**
     * @param chosenVertices   vertices to get a proposed change for
     * @param totalLogProbOld  The log of the previous state's probability
     * @param affectedVertices Downstream vertices of the chosenVertices
     * @param T                Temperature for simulated annealing. This
     *                         should be constant if no annealing is wanted
     * @param random           source of randomness
     * @return the log probability of the network after either accepting or rejecting the sample
     */
    static double nextSample(final Set<Vertex> chosenVertices,
                             final Map<Vertex, LambdaSection> affectedVertices,
                             final ProposalDistribution proposalDistribution,
                             final double totalLogProbOld,
                             final double T,
                             final KeanuRandom random) {

        final double affectedVerticesLogProbOld = sumLogProbabilityOfAffected(chosenVertices, affectedVertices);

        NetworkSnapshot preProposalSnapshot = getSnapshotOfAffectedVertices(chosenVertices, affectedVertices);

        Proposal proposal = proposalDistribution.getProposal(chosenVertices, random);
        proposal.apply();
        VertexValuePropagation.cascadeUpdate(chosenVertices);

        final double affectedVerticesLogProbNew = sumLogProbabilityOfAffected(chosenVertices, affectedVertices);

        if (affectedVerticesLogProbNew != LOG_ZERO_PROBABILITY) {

            final double totalLogProbNew = totalLogProbOld - affectedVerticesLogProbOld + affectedVerticesLogProbNew;

            final double pqxOld = logProbAtProposalFrom(chosenVertices, proposal);
            final double pqxNew = logProbAtProposalTo(chosenVertices, proposal);

            final double logR = (totalLogProbNew * (1.0 / T) + pqxOld) - (totalLogProbOld * (1.0 / T) + pqxNew);
            final double r = Math.exp(logR);

            final boolean shouldAccept = r >= random.nextDouble();

            if (shouldAccept) {
                return totalLogProbNew;
            }
        }

        proposal.reject();
        preProposalSnapshot.apply();

//        VertexValuePropagation.cascadeUpdate(chosenVertices);
        return totalLogProbOld;
    }

    static NetworkSnapshot getSnapshotOfAffectedVertices(final Set<Vertex> chosenVertices,
                                                         final Map<Vertex, LambdaSection> affectedVertices) {

        Set<Vertex> allAffectedVertices = new HashSet<>();
        for (Vertex vertex : chosenVertices) {
            allAffectedVertices.addAll(affectedVertices.get(vertex).getAllVertices());
        }

        return NetworkSnapshot.create(allAffectedVertices);
    }

    static Map<Vertex, LambdaSection> getVerticesAffectedByLatents(List<? extends Vertex> latentVertices,
                                                                   boolean includeNonProbabilistic) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> LambdaSection.getDownstreamLambdaSection(v, includeNonProbabilistic)
            ));
    }

    private static double sumLogProbabilityOfAffected(Set<Vertex> vertices,
                                                      Map<Vertex, LambdaSection> affectedVertices) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += sumLogProbability(affectedVertices.get(v).getProbabilisticVertices());
        }
        return sumLogProb;
    }

    /**
     * This returns the log probability of typically a subset of a bayesian network.
     *
     * @param vertices Vertices to consider in log probability calculation
     * @return the log probability of the set of vertices
     */
    private static double sumLogProbability(Set<Vertex> vertices) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += v.logProbAtValue();
        }
        return sumLogProb;
    }

    private static double logProbAtProposalFrom(Set<Vertex> vertices, Proposal proposal) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += v.logProb(proposal.getProposalFrom(v));
        }
        return sumLogProb;
    }

    private static double logProbAtProposalTo(Set<Vertex> vertices, Proposal proposal) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += v.logProb(proposal.getProposalTo(v));
        }
        return sumLogProb;
    }

    private static void takeSamples(Map<Long, List<?>> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Vertex<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }

    private static void checkBayesNetInHealthyState(BayesianNetwork bayesNet) {
        bayesNet.cascadeObservations();
        if (bayesNet.getLatentAndObservedVertices().isEmpty()) {
            throw new IllegalArgumentException("Cannot sample from a completely deterministic BayesNet");
        } else if (bayesNet.isInImpossibleState()) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

}
