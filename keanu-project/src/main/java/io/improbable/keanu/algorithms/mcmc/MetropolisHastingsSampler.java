package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.mcmc.proposal.MHStepVariableSelector;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public class MetropolisHastingsSampler implements SamplingAlgorithm {

    private final List<? extends Variable> latentVertices;
    private final List<? extends Variable> verticesToSampleFrom;
    private final MetropolisHastingsStep mhStep;
    private final MHStepVariableSelector variableSelector;

    private double logProbabilityBeforeStep;
    private int sampleNum;

    public MetropolisHastingsSampler(List<? extends Variable> latentVertices,
                                     List<? extends Variable> verticesToSampleFrom,
                                     MetropolisHastingsStep mhStep,
                                     MHStepVariableSelector variableSelector,
                                     double logProbabilityBeforeStep) {
        this.latentVertices = latentVertices;
        this.verticesToSampleFrom = verticesToSampleFrom;
        this.mhStep = mhStep;
        this.variableSelector = variableSelector;
        this.logProbabilityBeforeStep = logProbabilityBeforeStep;
        this.sampleNum = 0;
    }

    @Override
    public void step() {
        Set<Variable> chosenVertices = variableSelector.select(latentVertices, sampleNum);

        logProbabilityBeforeStep = mhStep.step(
            chosenVertices,
            logProbabilityBeforeStep
        ).getLogProbabilityAfterStep();

        sampleNum++;
    }

    @Override
    public void sample(Map<VariableReference, List<?>> samplesByVertex, List<Double> logOfMasterPForEachSample) {
        step();
        takeSamples(samplesByVertex, verticesToSampleFrom);
        logOfMasterPForEachSample.add(logProbabilityBeforeStep);
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(SamplingAlgorithm.takeSample((List<? extends Variable<Object>>) verticesToSampleFrom), logProbabilityBeforeStep);
    }
    private static void takeSamples(Map<VariableReference, List<?>> samples, List<? extends Variable> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex((Variable<?>) vertex, samples));
    }

    private static <T> void addSampleForVertex(Variable<T> vertex, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getReference(), v -> new ArrayList<T>());
        T value = vertex.getValue();
        samplesForVertex.add(value);
        log.trace(String.format("Sampled %s", value));
    }

}

