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
    private int N = 100000;
    private double epsilon = 0.01;
    private Integer lowerBound = 10;
    private Integer upperBound = 20;
    private List<Integer> samples = new ArrayList<>();
    private Random random;

    @Before
    public void setup() {

        random = new Random(1);

        UniformIntVertex testUniformVertex = new UniformIntVertex(
                new ConstantIntegerVertex(lowerBound),
                new ConstantIntegerVertex(upperBound)
        );

        for (int i = 0; i < N; i++) {
            Integer sample = testUniformVertex.sample(random);
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
