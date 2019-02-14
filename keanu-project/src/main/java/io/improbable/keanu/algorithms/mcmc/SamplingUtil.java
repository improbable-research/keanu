package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SamplingUtil {

    public static void takeSamples(Map<VariableReference, List<?>> samples, List<? extends Variable> fromVariables) {
        fromVariables.forEach(variable -> addSampleForVariable((Variable<?, ?>) variable, samples));
    }

    private static <T> void addSampleForVariable(Variable<T, ?> variable, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVariable = (List<T>) samples.computeIfAbsent(variable.getReference(), v -> new ArrayList<T>());
        T value = variable.getValue();
        samplesForVariable.add(value);
    }

}
