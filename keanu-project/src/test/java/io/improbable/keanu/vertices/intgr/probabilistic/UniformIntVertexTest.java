package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;

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
    UniformIntVertex testUniformVertex =
        new UniformIntVertex(new int[] {1, N}, lowerBound, upperBound);
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
    assertTrue(maxSample < upperBound);
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
    UniformIntVertex testUniformVertex =
        new UniformIntVertex(new int[] {1, 100}, Integer.MIN_VALUE, Integer.MAX_VALUE);
    IntegerTensor sample = testUniformVertex.sample(random);

    Set<Integer> uniqueValues = new HashSet<>(sample.asFlatList());

    assertTrue(uniqueValues.size() > 1);
  }

  @Test
  public void logProbUpperBoundIsNegativeInfinity() {
    UniformIntVertex testUniformVertex =
        new UniformIntVertex(new int[] {1, N}, lowerBound, upperBound);
    assertEquals(
        testUniformVertex.logProb(Nd4jIntegerTensor.scalar(upperBound)),
        Double.NEGATIVE_INFINITY,
        1e-6);
  }

  @Test
  public void logProbLowerBoundIsNotNegativeInfinity() {
    UniformIntVertex testUniformVertex =
        new UniformIntVertex(new int[] {1, N}, lowerBound, upperBound);
    assertNotEquals(
        testUniformVertex.logProb(Nd4jIntegerTensor.scalar(lowerBound)),
        Double.NEGATIVE_INFINITY,
        1e-6);
  }
}
