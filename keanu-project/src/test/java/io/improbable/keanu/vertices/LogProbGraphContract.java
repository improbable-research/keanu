package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static org.hamcrest.Matchers.equalTo;
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

    public static void equalTensor(LogProbGraph logProbGraph, DoubleTensor expectedLogDensityTensor) {
        DoubleTensor logProbGraphOutputTensor = logProbGraph.getLogProbOutput().getValue();
        assertThat(expectedLogDensityTensor, TensorMatchers.valuesAndShapesMatch(logProbGraphOutputTensor));
    }
}
