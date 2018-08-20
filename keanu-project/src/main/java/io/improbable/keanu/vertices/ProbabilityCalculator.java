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
}
