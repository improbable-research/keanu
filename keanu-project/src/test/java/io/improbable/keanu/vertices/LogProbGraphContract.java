package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class LogProbGraphContract {

    public static void matchesKnownLogDensity(LogProbGraph logProbGraph, double expectedLogDensity) {
        DoubleVertex logProbGraphOutput = logProbGraph.getLogProbOutput();
        double actualDensity = logProbGraphOutput.getValue().sum();
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }

    public static void equal(LogProbGraph actual, LogProbGraph expected) {
        assertThat(actual.getLogProbOutput().getValue(), equalTo(expected.getLogProbOutput().getValue()));
    }

    public static void equalFlatArray(LogProbGraph logProbGraph, double[] flatArray) {
        DoubleVertex logProbGraphOutput = logProbGraph.getLogProbOutput();
        assertArrayEquals(logProbGraphOutput.getValue().asFlatDoubleArray(), flatArray, 1e-5);
    }
}
