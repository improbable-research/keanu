package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class GaussianProposalDistribution implements ProposalDistribution {

    private final DoubleTensor sigma;
    private final List<ProposalListener> listeners;

    public GaussianProposalDistribution(DoubleTensor sigma) {
        this(sigma, Collections.emptyList());
    }

    public GaussianProposalDistribution(DoubleTensor sigma, List<ProposalListener> listeners) {
        this.sigma = sigma;
        this.listeners = listeners;
    }

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        Proposal proposal = new Proposal();
        proposal.addListeners(listeners);
        for (Vertex vertex : vertices) {
            ContinuousDistribution proposalDistribution = Gaussian.withParameters((DoubleTensor) vertex.getValue(), sigma);
            proposal.setProposal(vertex, proposalDistribution.sample(vertex.getShape(), random));
        }
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> vertex, T ofValue, T givenValue) {
        if (!(ofValue instanceof DoubleTensor)) {
            throw new ClassCastException("Only DoubleTensor values are supported - not " + ofValue.getClass().getSimpleName());
        }

        ContinuousDistribution proposalDistribution = Gaussian.withParameters((DoubleTensor) ofValue, sigma);
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

}
