package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

public class UniformVertexTest {
  private int N = 100000;
  private Double lowerBound = 10.;
  private Double upperBound = 20.;
  private List<Double> samples = new ArrayList<>();
  private KeanuRandom random;

  @Before
  public void setup() {
    random = new KeanuRandom(1);
    UniformVertex testUniformVertex = new UniformVertex(new int[] {1, N}, lowerBound, upperBound);
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
    UniformVertex testUniformVertex =
        new UniformVertex(new int[] {1, 100}, Double.MIN_VALUE, Double.MAX_VALUE);
    DoubleTensor sample = testUniformVertex.sample(random);

    Set<Double> uniqueValues = new HashSet<>(sample.asFlatList());

    assertTrue(uniqueValues.size() > 1);
  }

  @Test
  public void logProbUpperBoundIsNegativeInfinity() {
    UniformVertex testUniformVertex = new UniformVertex(new int[] {1, N}, lowerBound, upperBound);
    assertEquals(
        testUniformVertex.logProb(Nd4jDoubleTensor.scalar(upperBound)),
        Double.NEGATIVE_INFINITY,
        1e-6);
  }

  @Test
  public void logProbLowerBoundIsNotNegativeInfinity() {
    UniformVertex testUniformVertex = new UniformVertex(new int[] {1, N}, lowerBound, upperBound);
    assertNotEquals(
        testUniformVertex.logProb(Nd4jDoubleTensor.scalar(lowerBound)),
        Double.NEGATIVE_INFINITY,
        1e-6);
  }

  @Test
  public void uniformSampleMethodMatchesLogProbMethod() {
    UniformVertex testUniformVertex =
        new UniformVertex(
            new int[] {1, N}, ConstantVertex.of(lowerBound), ConstantVertex.of(upperBound));
    ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(
        testUniformVertex, lowerBound, upperBound - 1, 0.5, 1e-2, random);
  }
}
