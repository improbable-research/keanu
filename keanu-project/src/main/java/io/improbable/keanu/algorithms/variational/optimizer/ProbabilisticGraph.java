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

    List<? extends Variable> getLatentVariables();

    /**
     * Tells you if a value of log probability is impossible, i.e. -Infinity.
     * It also includes the case where it is NaN.
     * @param logProb log probability, a value in the range [-Infinity, 0]
     * @return true or false
     */
    static boolean isImpossible(double logProb) {
        return logProb == Double.NEGATIVE_INFINITY || Double.isNaN(logProb);
    }
}
