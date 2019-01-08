package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import static org.junit.Assert.assertEquals;

public class LogProbGraphContract {
    public static void matchesKnownLogDensity(LogProbGraph logProbGraph, double expectedLogDensity) {

        DoubleVertex logProbGraphOutput = logProbGraph.getLogProbOutput();
        double actualDensity = logProbGraphOutput.getValue().sum();
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }
}
