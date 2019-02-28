package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Assumes variables are independent. Instead of using a Multivariate Gaussian pdf,
 * this proposal computes the product of multiple Univariate Gaussian pdfs.
 */
public class MultivariateGaussianProposalDistribution implements ProposalDistribution {

    private final Map<? extends Variable, DoubleTensor> sigmas;
    private final ProposalNotifier proposalNotifier;

    public MultivariateGaussianProposalDistribution(Map<? extends Variable, DoubleTensor> sigmas) {
        this(sigmas, Collections.emptyList());
    }

    public MultivariateGaussianProposalDistribution(Map<? extends Variable, DoubleTensor> sigmas, List<ProposalListener> listeners) {
        this.sigmas = sigmas;
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable variable : variables) {
            if (!(variable.getValue() instanceof DoubleTensor)) {
                throw new IllegalStateException("Multivariate Gaussian proposal function cannot be used for discrete variable " + variable);
            }
            DoubleTensor sample = random.nextGaussian(variable.getShape(), (DoubleTensor) variable.getValue(), sigmas.get(variable));
            proposal.setProposal(variable, sample);
        }
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        if (!(ofValue instanceof DoubleTensor)) {
            throw new ClassCastException("Only DoubleTensor values are supported - not " + ofValue.getClass().getSimpleName());
        }

        ContinuousDistribution proposalDistribution = Gaussian.withParameters((DoubleTensor) ofValue, sigmas.get(variable));
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }
}
