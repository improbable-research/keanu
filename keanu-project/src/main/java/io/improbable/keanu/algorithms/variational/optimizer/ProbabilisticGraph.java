package io.improbable.keanu.algorithms.variational.optimizer;

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

    List<VariableReference> getLatentVariables();

    Map<VariableReference, ?> getLatentVariablesValues();

    Map<VariableReference, long[]> getLatentVariablesShapes();
}
