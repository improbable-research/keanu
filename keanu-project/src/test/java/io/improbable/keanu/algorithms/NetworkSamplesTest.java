package io.improbable.keanu.algorithms;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.VertexId;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertTrue;

public class NetworkSamplesTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    NetworkSamples samples;
    VertexId v1 = new VertexId();
    VertexId v2 = new VertexId();

    @Before
    public void setup() {

        Map<VariableReference, List<Integer>> sampleMap = new HashMap<>();
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

    @Category(Slow.class)
    @Test
    public void doesSubsample() {
        NetworkSamples subsamples = samples.downSample(5);

        assertTrue(subsamples.size() == 2);
        assertTrue(subsamples.get(v1).asList().equals(Arrays.asList(1, 6)));
        assertTrue(subsamples.get(v2).asList().equals(Arrays.asList(9, 4)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesCatchInvalidDropCount() {
        samples.drop(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesCatchInvalidDownSampleInterval() {
        samples.downSample(0);
    }

    @Test
    public void doesNothingOnDropZero() {
        assertEquals(samples, samples.drop(0));
    }

    @Test
    public void doesCalculateProbability() {
        double result2 = samples.probability(state -> {
            int a = state.get(v1);
            int b = state.get(v2);
            return a == b;
        });
        assertTrue(result2 == 0.1);
    }

    @Test
    public void doesFind100PercentProbability() {

        double result = samples.probability(state -> {
            int a = state.get(v1);
            int b = state.get(v2);
            return (a + b) == 10;
        });
        assertTrue(result == 1.0);

    }

    @Test
    public void youCanGetTheMostProbableState() {
        NetworkState mostProbableState = samples.getMostProbableState();
        assertThat(mostProbableState.get(v1), equalTo(10));
        assertThat(mostProbableState.get(v2), equalTo(0));
    }

    @Test
    public void canBeConstructedFromListOfNetworkSample() {
        List<Double> v1Samples = Arrays.asList(33.2, 3.9);
        List<Double> v2Samples = Arrays.asList(109.4, 3.55);
        final List<Double> logOfMasterPBySample = Arrays.asList(9.4, 12.7);
        Map<VariableReference, Double> vertexValsFirstSample = ImmutableMap.of(
            v1, v1Samples.get(0),
            v2, v2Samples.get(0)
        );
        Map<VariableReference, Double> vertexValsSecondSample = ImmutableMap.of(
            v1, v1Samples.get(1),
            v2, v2Samples.get(1)
        );
        NetworkSamples networkSamples = NetworkSamples.from(ImmutableList.of(
            new NetworkSample(vertexValsFirstSample, logOfMasterPBySample.get(0)),
            new NetworkSample(vertexValsSecondSample, logOfMasterPBySample.get(1))
        ));
        assertEquals(v1Samples, networkSamples.get(v1).asList());
        assertEquals(v2Samples, networkSamples.get(v2).asList());
        assertEquals(logOfMasterPBySample.get(0), networkSamples.getLogOfMasterP(0));
        assertEquals(logOfMasterPBySample.get(1), networkSamples.getLogOfMasterP(1));
    }
}
