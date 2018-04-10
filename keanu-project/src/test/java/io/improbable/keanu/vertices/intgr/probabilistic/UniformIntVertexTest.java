package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

public class UniformIntVertexTest {
    int N = 100000;
    double epsilon = 0.01;
    Integer lowerBound = 10;
    Integer upperBound = 20;
    List<Integer> samples = new ArrayList<>();

    @Before
    public void setup() {
        UniformIntVertex testUniformVertex = new UniformIntVertex(
                new ConstantIntegerVertex(lowerBound),
                new ConstantIntegerVertex(upperBound),
                new Random(1));

        for (int i = 0; i < N; i++) {
            Integer sample = testUniformVertex.sample();
            samples.add(sample);
        }
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
}
