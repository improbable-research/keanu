package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.*;

public class GaussianProposalDistribution implements ProposalDistribution {

    private final Map<? extends Variable, DoubleTensor> sigmas;
    private final ProposalNotifier proposalNotifier;

    public GaussianProposalDistribution(List<? extends Variable> variables, DoubleTensor sigma) {
        this(toSigmasMap(variables, sigma));
    }

    private static Map<? extends Variable, DoubleTensor> toSigmasMap(Collection<? extends Variable> variables, DoubleTensor sigma) {
        Map<Variable, DoubleTensor> sigmasMap = new HashMap<>();
        for (Variable variable : variables) {
            sigmasMap.put(variable, sigma);
        }
        return sigmasMap;
    }

    public GaussianProposalDistribution(Map<? extends Variable, DoubleTensor> sigmas) {
        this(sigmas, Collections.emptyList());
    }

    public GaussianProposalDistribution(Map<? extends Variable, DoubleTensor> sigmas, List<ProposalListener> listeners) {
        this.sigmas = sigmas;
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable variable : variables) {
            if (!(variable.getValue() instanceof DoubleTensor)) {
                throw new IllegalStateException("Gaussian proposal function cannot be used for discrete variable " + variable);
            }
            DoubleTensor sample = random.nextGaussian(variable.getShape(), (DoubleTensor) variable.getValue(), sigmas.get(variable));
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

        Gaussian proposalDistribution = (Gaussian) Gaussian.withParameters((DoubleTensor) ofValue, sigmas.get(variable));
        return proposalDistribution.logProb((DoubleTensor) givenValue).sum();
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }
}
