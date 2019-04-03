package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class GaussianProposalDistribution implements ProposalDistribution {

    private final Map<? extends Variable, DoubleTensor> sigmas;
    private final DoubleTensor defaultSigma;
    private final ProposalNotifier proposalNotifier;

    private GaussianProposalDistribution(Map<? extends Variable, DoubleTensor> sigmas, double defaultSigma, ProposalNotifier proposalNotifier) {
        this.sigmas = sigmas;
        this.defaultSigma = DoubleTensor.scalar(defaultSigma);
        this.proposalNotifier = proposalNotifier;
    }

    public static GaussianProposalDistributionBuilder builder() {
        return new GaussianProposalDistributionBuilder();
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable variable : variables) {
            if (!(variable.getValue() instanceof DoubleTensor)) {
                throw new IllegalStateException("Gaussian proposal function cannot be used for discrete variable " + variable);
            }

            DoubleTensor sample = random.nextGaussian(variable.getShape(), (DoubleTensor) variable.getValue(), sigmas.getOrDefault(variable, defaultSigma));
            proposal.setProposal(variable, sample);
        }
        proposalNotifier.notifyProposalCreated(proposal);
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        if (!(ofValue instanceof DoubleTensor)) {
            throw new ClassCastException("Only DoubleTensor values are supported - not " + ofValue.getClass().getSimpleName());
        }

        Gaussian proposalDistribution = (Gaussian) Gaussian.withParameters((DoubleTensor) ofValue, sigmas.getOrDefault(variable, defaultSigma));
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }

    public static class GaussianProposalDistributionBuilder {

        private Map<Variable, DoubleTensor> sigmas = new HashMap<>();
        private double defaultSigma = 1.0;
        private ProposalNotifier proposalNotifier = new ProposalNotifier(Collections.emptyList());

        private GaussianProposalDistributionBuilder() {
        }

        public GaussianProposalDistributionBuilder sigma(Variable sigmaKey, DoubleTensor sigmaValue) {
            sigmas.put(sigmaKey, sigmaValue);
            return this;
        }

        public GaussianProposalDistributionBuilder defaultSigma(double defaultSigma) {
            this.defaultSigma = defaultSigma;
            return this;
        }

        public GaussianProposalDistributionBuilder proposalNotifier(ProposalNotifier proposalNotifier) {
            this.proposalNotifier = proposalNotifier;
            return this;
        }

        public GaussianProposalDistribution build() {
            return new GaussianProposalDistribution(sigmas, defaultSigma, proposalNotifier);
        }
    }
}
