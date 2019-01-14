package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.mcmc.MetropolisHastingsSampler;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastingsStep;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.network.NetworkSnapshot;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    /**
     * An optimisation on top of logProb.
     *
     * This method can be used to save computation by computing log prob only
     * for the variables in the graph that are affected (down stream) of the provided variables.
     *
     * This defaults to calling logProb().
     *
     * @param variables the variables from which we calculate the logProb of each of their downstream variables
     * @return log prob of the affected variables
     */
    default double downstreamLogProb(Set<? extends Variable> variables) {
        return logProb();
    }

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor>> getContinuousLatentVariables();

    void cascadeUpdate(Set<? extends Variable> inputs);

    void cascadeFixedVariables();

    NetworkSnapshot getSnapshotOfAllAffectedVariables(Set<? extends Variable> variables);

    MetropolisHastingsSampler metropolisHastingsSampler(List<? extends Variable> verticesToSampleFrom, MetropolisHastingsStep mhStep, MHStepVariableSelector variableSelector);
}
