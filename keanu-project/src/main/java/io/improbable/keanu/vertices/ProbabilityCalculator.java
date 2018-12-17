package io.improbable.keanu.vertices;

import java.util.Collection;

public class ProbabilityCalculator {
    private ProbabilityCalculator() {}

    public static double calculateLogProbFor(Collection<? extends Vertex> vertices) {
        double sum = 0.0;
        for (Vertex<?> vertex : vertices) {
            if (vertex instanceof Probabilistic) {
                sum += ((Probabilistic) vertex).logProbAtValue();
            } else if (vertex instanceof NonProbabilistic) {
                if (((NonProbabilistic) vertex).contradictsObservation()) {
                    return Double.NEGATIVE_INFINITY;
                }
            } else {
                throw new IllegalArgumentException("Found a vertex that is neither Probabilistic nor Non-Probabilistic");
            }
        }
        return sum;
    }

    /**
     * Tells you if a value of log probability is impossible, i.e. -Infinity.
     * It also includes the case where it is NaN.
     * @param logProb log probability, a value in the range [-Infinity, 0]
     * @return true or false
     */
    public static boolean isImpossibleLogProb(double logProb) {
        return logProb == Double.NEGATIVE_INFINITY || Double.isNaN(logProb);
    }
}
