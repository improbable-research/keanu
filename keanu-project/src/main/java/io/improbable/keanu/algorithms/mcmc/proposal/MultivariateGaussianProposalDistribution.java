package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class MultivariateGaussianProposalDistribution implements ProposalDistribution {

    private final DoubleTensor covariance;
    private final DoubleTensor cholesky;
    private final ProposalNotifier proposalNotifier;

    public MultivariateGaussianProposalDistribution(DoubleTensor covariance) {
        this(covariance, Collections.emptyList());
    }

    public MultivariateGaussianProposalDistribution(DoubleTensor covariance, List<ProposalListener> listeners) {
        if (covariance.getRank() != 2) {
            throw new IllegalArgumentException("covariance for Multivariate Gaussian proposal function has to me a matrix.");
        }
        if (covariance.getShape()[0] != covariance.getShape()[1]) {
            throw new IllegalArgumentException("covariance matrix for Multivariate Gaussian proposal has to be symmetric.");
        }

        this.covariance = covariance;
        this.cholesky = covariance.choleskyDecomposition();
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable variable : variables) {
            if (!(variable.getValue() instanceof DoubleTensor)) {
                throw new IllegalStateException("Multivariate Gaussian proposal function cannot be used for discrete variable " + variable);
            }
            DoubleTensor sample = cholesky
                .matrixMultiply(random.nextGaussian(new long[]{covariance.getShape()[0]}))
                .reshape(variable.getShape());
            proposal.setProposal(variable, sample);
        }
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        if (!(ofValue instanceof DoubleTensor)) {
            throw new ClassCastException("Only DoubleTensor values are supported - not " + ofValue.getClass().getSimpleName());
        }

        ContinuousDistribution proposalDistribution = MultivariateGaussian.withParameters((DoubleTensor) ofValue, covariance);
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }
}
