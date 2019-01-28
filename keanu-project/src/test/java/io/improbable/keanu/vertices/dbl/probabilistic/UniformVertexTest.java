package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

public class UniformVertexTest {
    private int N = 100000;
    private Double lowerBound = 10.;
    private Double upperBound = 20.;
    private List<Double> samples = new ArrayList<>();
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        UniformVertex testUniformVertex = new UniformVertex(new long[]{1, N}, lowerBound, upperBound);
        samples.addAll(testUniformVertex.sample(random).asFlatList());
    }

    @Test
    public void allSamplesAreWithinBounds() {
        Double minSample = Collections.min(samples);
        Double maxSample = Collections.max(samples);

        assertTrue(minSample >= lowerBound);
        assertTrue(maxSample < upperBound);
    }

    @Test
    public void exclusiveUpperBoundIsNeverProduced() {
        assertFalse(samples.contains(upperBound));
    }

    @Test
    public void canUseFullDoubleRange() {
        UniformVertex testUniformVertex = new UniformVertex(new long[]{1, 100}, Double.MIN_VALUE, Double.MAX_VALUE);
        DoubleTensor sample = testUniformVertex.sample(random);

        Set<Double> uniqueValues = new HashSet<>(sample.asFlatList());

        assertTrue(uniqueValues.size() > 1);
    }

    @Test
    public void logProbUpperBoundIsNegativeInfinity() {
        UniformVertex testUniformVertex = new UniformVertex(new long[]{1, N}, lowerBound, upperBound);
        assertEquals(testUniformVertex.logProb(Nd4jDoubleTensor.scalar(upperBound)), Double.NEGATIVE_INFINITY, 1e-6);
    }

    @Test
    public void logProbGraphUpperBoundIsNegativeInfinity() {
        DoubleVertex xMin = ConstantVertex.of(lowerBound);
        DoubleVertex xMax = ConstantVertex.of(upperBound);
        UniformVertex uniformVertex = new UniformVertex(xMin, xMax);
        LogProbGraph logProbGraph = uniformVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, uniformVertex, DoubleTensor.scalar(upperBound));

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, Double.NEGATIVE_INFINITY);
    }

    @Test
    public void logProbLowerBoundIsNotNegativeInfinity() {
        UniformVertex testUniformVertex = new UniformVertex(new long[]{1, N}, lowerBound, upperBound);
        assertNotEquals(testUniformVertex.logProb(Nd4jDoubleTensor.scalar(lowerBound)), Double.NEGATIVE_INFINITY, 1e-6);
    }

    @Test
    public void logProbGraphLowerBoundIsNotNegativeInfinity() {
        DoubleVertex xMin = ConstantVertex.of(lowerBound);
        DoubleVertex xMax = ConstantVertex.of(upperBound);
        UniformVertex uniformVertex = new UniformVertex(xMin, xMax);
        LogProbGraph logProbGraph = uniformVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, xMin, xMin.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, xMax, xMax.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, uniformVertex, DoubleTensor.scalar(lowerBound));

        UniformRealDistribution uniformRealDistribution = new UniformRealDistribution(lowerBound, upperBound);
        double expectedDensity = uniformRealDistribution.logDensity(lowerBound);

        assertThat(expectedDensity, not(equalTo(Double.NEGATIVE_INFINITY)));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Category(Slow.class)
    @Test
    public void uniformSampleMethodMatchesLogProbMethod() {
        UniformVertex testUniformVertex = new UniformVertex(new long[] {1, N}, ConstantVertex.of(lowerBound), ConstantVertex.of(upperBound));
        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(testUniformVertex, lowerBound, upperBound - 1, 0.5, 1e-2, random);
    }
}
