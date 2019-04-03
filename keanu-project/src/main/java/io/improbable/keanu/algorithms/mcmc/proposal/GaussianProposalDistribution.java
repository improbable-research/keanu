package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import lombok.Builder;
import lombok.NonNull;
import lombok.Singular;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

@Builder
public class GaussianProposalDistribution implements ProposalDistribution {

    @Singular
    @NonNull
    private final Map<? extends Variable, DoubleTensor> sigmas;

    @NonNull
    @Builder.Default
    private final DoubleTensor defaultSigma = DoubleTensor.scalar(1.0);

    @NonNull
    @Builder.Default
    private final ProposalNotifier proposalNotifier = new ProposalNotifier(Collections.emptyList());

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
}
