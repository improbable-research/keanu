package io.improbable.keanu.algorithms.mcmc.proposal;

import java.util.Set;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class GaussianProposalDistribution implements ProposalDistribution {

    private final DoubleTensor sigma;

    public GaussianProposalDistribution(DoubleTensor sigma) {
        this.sigma = sigma;
    }

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Vertex vertex : vertices) {
            Gaussian proposalDistribution = Gaussian.withParameters((DoubleTensor) vertex.getValue(), sigma);
            proposal.setProposal(vertex, proposalDistribution.sample(vertex.getShape(), random));
        }
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> vertex, T ofValue, T givenValue) {
        if (!(ofValue instanceof DoubleTensor)) {
            throw new ClassCastException("Only DoubleTensor values are supported - not " + ofValue.getClass().getSimpleName());
        }

        Gaussian proposalDistribution = Gaussian.withParameters((DoubleTensor) ofValue, sigma);
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

}
