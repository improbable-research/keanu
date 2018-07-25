package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;

public class UniformIntVertexTest {
    private int N = 100000;
    private double epsilon = 0.05;
    private Integer lowerBound = 10;
    private Integer upperBound = 20;
    private List<Integer> samples = new ArrayList<>();
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        UniformIntVertex testUniformVertex = new DistributionVertexBuilder()
            .shaped(1, N)
            .withInput(ParameterName.MIN, lowerBound)
            .withInput(ParameterName.MAX, upperBound)
            .uniformInt();
        samples.addAll(testUniformVertex.sample(random).asFlatList());
    }

    @Test
    public void samplingProducesRealisticMean() {
        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        Double expectedMean = ((double) lowerBound + (double) upperBound - 1) / 2.0;

        double mean = stats.getMean();
        assertEquals(mean, expectedMean, epsilon);
    }

    @Test
    public void allSamplesAreWithinBounds() {
        Integer minSample = Collections.min(samples);
        Integer maxSample = Collections.max(samples);

        assertTrue(minSample >= lowerBound);
        assertTrue(maxSample <= upperBound);
    }

    @Test
    public void allSamplesAreProducedAtLeastOnce() {
        assertTrue(samples.contains(10));
        assertTrue(samples.contains(11));
        assertTrue(samples.contains(12));
        assertTrue(samples.contains(13));
        assertTrue(samples.contains(14));
        assertTrue(samples.contains(15));
        assertTrue(samples.contains(16));
        assertTrue(samples.contains(17));
        assertTrue(samples.contains(18));
        assertTrue(samples.contains(19));
    }

    @Test
    public void exclusiveUpperBoundIsNeverProduced() {
        assertFalse(samples.contains(upperBound));
    }

    @Test
    public void canUseFullIntRange() {
        UniformIntVertex testUniformVertex = new DistributionVertexBuilder()
            .shaped(1, 100)
            .withInput(ParameterName.MIN, Integer.MIN_VALUE)
            .withInput(ParameterName.MAX, Integer.MAX_VALUE)
            .uniformInt();
        IntegerTensor sample = testUniformVertex.sample(random);

        Set<Integer> uniqueValues = new HashSet<>(sample.asFlatList());

        assertTrue(uniqueValues.size() > 1);
    }
}
