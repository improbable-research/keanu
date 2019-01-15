package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.mcmc.MetropolisHastingsSampler;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastingsStep;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor>> getContinuousLatentVariables();

    MetropolisHastingsSampler metropolisHastingsSampler(List<? extends Variable> verticesToSampleFrom, MetropolisHastingsStep mhStep, MHStepVariableSelector variableSelector);
}
