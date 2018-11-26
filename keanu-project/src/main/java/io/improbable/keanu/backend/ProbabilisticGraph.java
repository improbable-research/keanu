package io.improbable.keanu.backend;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<String, ?> inputs);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<String, ?> inputs);

    List<String> getLatentVariables();

    Map<String, ?> getLatentVariablesValues();

    Map<String, long[]> getLatentVariablesShapes();
}
