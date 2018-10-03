package io.improbable.keanu.algorithms;

import static org.junit.Assert.assertTrue;

import io.improbable.keanu.vertices.VertexId;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;

public class NetworkSamplesTest {

    NetworkSamples samples;
    VertexId v1 = new VertexId();
    VertexId v2 = new VertexId();

    @Before
    public void setup() {

        Map<VertexId, List<Integer>> sampleMap = new HashMap<>();
        sampleMap.put(v1, Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        sampleMap.put(v2, Arrays.asList(9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

        List<Double> logProbs = Arrays.asList(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.);

        samples = new NetworkSamples(sampleMap, logProbs, 10);
    }

    @Test
    public void doesDropSamples() {
        NetworkSamples droppedSamples = samples.drop(5);

        assertTrue(droppedSamples.size() == 5);
        assertTrue(droppedSamples.get(v1).asList().equals(Arrays.asList(6, 7, 8, 9, 10)));
        assertTrue(droppedSamples.get(v2).asList().equals(Arrays.asList(4, 3, 2, 1, 0)));
    }

    @Test
    public void doesSubsample() {
        NetworkSamples subsamples = samples.downSample(5);

        assertTrue(subsamples.size() == 2);
        assertTrue(subsamples.get(v1).asList().equals(Arrays.asList(1, 6)));
        assertTrue(subsamples.get(v2).asList().equals(Arrays.asList(9, 4)));
    }

    @Test
    public void doesCalculateProbability() {
        double result2 =
                samples.probability(
                        state -> {
                            int a = state.get(v1);
                            int b = state.get(v2);
                            return a == b;
                        });
        assertTrue(result2 == 0.1);
    }

    @Test
    public void doesFind100PercentProbability() {

        double result =
                samples.probability(
                        state -> {
                            int a = state.get(v1);
                            int b = state.get(v2);
                            return (a + b) == 10;
                        });
        assertTrue(result == 1.0);
    }
}
