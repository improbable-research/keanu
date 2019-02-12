package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public interface SamplingAlgorithm {

    static Map<VariableReference, ?> takeSample(List<? extends Variable> sampleFromVariables) {
        return sampleFromVariables.stream()
            .collect(Collectors.toMap(Variable::getReference, Variable::getValue));
    }

    static void takeSamples(Map<VariableReference, List<?>> samples, List<? extends Variable> fromVariables) {
        fromVariables.forEach(variable -> addSampleForVariable((Variable<?, ?>) variable, samples));
    }

    static <T> void addSampleForVariable(Variable<T, ?> variable, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVariable = (List<T>) samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<T>());
        T value = variable.getValue();
        samplesForVariable.add(value);
    }

    /**
     * Move forward the state of the Sampling Algorithm by a single step but do not return anything.
     */
    void step();

    /**
     * Takes a sample with the algorithm and saves it in the supplied map (creating a new entry in the list if the
     * Variable already exists).
     *
     * @param samples                   map to store sampled variable values
     * @param logOfMasterPForEachSample list of log of master probability for each sample
     */
    void sample(Map<VariableReference, List<?>> samples, List<Double> logOfMasterPForEachSample);

    /**
     * Takes a sample with the algorithm and returns the state of the network for that sample.
     *
     * @return a network state that represents the current state of the algorithm.
     */
    NetworkSample sample();
}
