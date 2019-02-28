package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MultivariateGaussianProposalDistribution implements ProposalDistribution {

    private final Map<VariableReference, Double> sigmas;
    private final DoubleTensor covariance;
    private final ProposalNotifier proposalNotifier;

    public MultivariateGaussianProposalDistribution(Map<VariableReference, Double> sigmas) {
        this(sigmas, Collections.emptyList());
    }

    public MultivariateGaussianProposalDistribution(Map<VariableReference, Double> sigmas, List<ProposalListener> listeners) {
        this.sigmas = sigmas;
        this.covariance = DoubleTensor.create(sigmas.values().stream().mapToDouble(Double::doubleValue).toArray());
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable variable : variables) {
            if (!(variable.getValue() instanceof DoubleTensor)) {
                throw new IllegalStateException("Multivariate Gaussian proposal function cannot be used for discrete variable " + variable);
            }
            DoubleTensor sample = random.nextGaussian(variable.getShape(), (DoubleTensor) variable.getValue(), DoubleTensor.scalar(sigmas.get(variable.getReference())));
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
