package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface ProbabilisticModel {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    double logProb(Proposal proposal);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor, ?>> getContinuousLatentVariables();
}
