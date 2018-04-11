package io.improbable.keanu.algorithms;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class NetworkSamplesTest {

    NetworkSamples samples;

    @Before
    public void setup() {

        Map<String, List<Integer>> sampleMap = new HashMap<>();
        sampleMap.put("A", Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        sampleMap.put("B", Arrays.asList(9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

        samples = new NetworkSamples(sampleMap, 10);
    }

    @Test
    public void doesDropSamples() {
        NetworkSamples droppedSamples = samples.drop(5);

        assertTrue(droppedSamples.size() == 5);
        assertTrue(droppedSamples.get("A").asList().equals(Arrays.asList(6, 7, 8, 9, 10)));
        assertTrue(droppedSamples.get("B").asList().equals(Arrays.asList(4, 3, 2, 1, 0)));
    }

    @Test
    public void doesSubsample() {
        NetworkSamples subsamples = samples.downSample(5);

        assertTrue(subsamples.size() == 2);
        assertTrue(subsamples.get("A").asList().equals(Arrays.asList(1, 6)));
        assertTrue(subsamples.get("B").asList().equals(Arrays.asList(9, 4)));
    }

    @Test
    public void doesCalculateProbability() {
        double result2 = samples.probability(state -> {
            int a = state.get("A");
            int b = state.get("B");
            return a == b;
        });
        assertTrue(result2 == 0.1);
    }

    @Test
    public void doesFind100PercentProbability() {

        double result = samples.probability(state -> {
            int a = state.get("A");
            int b = state.get("B");
            return (a + b) == 10;
        });
        assertTrue(result == 1.0);

    }
}
