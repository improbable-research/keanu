package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public class MetropolisHastingsSampler implements SamplingAlgorithm {

    private final List<? extends Variable> latentVariables;
    private final List<? extends Variable> variablesToSampleFrom;
    private final MetropolisHastingsStep mhStep;
    private final MHStepVariableSelector variableSelector;

    private double logProbabilityBeforeStep;
    private int sampleNum;

    public MetropolisHastingsSampler(List<? extends Variable> latentVariables,
                                     List<? extends Variable> variablesToSampleFrom,
                                     MetropolisHastingsStep mhStep,
                                     MHStepVariableSelector variableSelector,
                                     double logProbabilityBeforeStep) {
        this.latentVariables = latentVariables;
        this.variablesToSampleFrom = variablesToSampleFrom;
        this.mhStep = mhStep;
        this.variableSelector = variableSelector;
        this.logProbabilityBeforeStep = logProbabilityBeforeStep;
        this.sampleNum = 0;
    }

    @Override
    public void step() {
        Set<Variable> chosenVariables = variableSelector.select(latentVariables, sampleNum);

        logProbabilityBeforeStep = mhStep.step(
            chosenVariables,
            logProbabilityBeforeStep
        ).getLogProbabilityAfterStep();

        sampleNum++;
    }

    @Override
    public void sample(Map<VariableReference, List<?>> samplesByVariable, List<Double> logOfMasterPForEachSample) {
        step();
        SamplingAlgorithm.takeSamples(samplesByVariable, variablesToSampleFrom);
        logOfMasterPForEachSample.add(logProbabilityBeforeStep);
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(SamplingAlgorithm.takeSample((List<? extends Variable<Object, ?>>) variablesToSampleFrom), logProbabilityBeforeStep);
    }

}
