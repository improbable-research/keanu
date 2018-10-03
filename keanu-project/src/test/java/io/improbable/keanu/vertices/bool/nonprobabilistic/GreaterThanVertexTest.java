package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static org.junit.Assert.assertEquals;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import org.junit.Test;

public class GreaterThanVertexTest {

  @Test
  public void comparesIntegers() {
    isGreaterThan(0, 1, false);
    isGreaterThan(1, 1, false);
    isGreaterThan(2, 1, true);
  }

  @Test
  public void comparesDoubles() {
    isGreaterThan(0.0, 0.5, false);
    isGreaterThan(0.5, 0.5, false);
    isGreaterThan(1.0, 0.5, true);
  }

  private void isGreaterThan(int a, int b, boolean expected) {
    BoolVertex vertex = ConstantVertex.of(a).greaterThan(ConstantVertex.of(b));
    assertEquals(expected, vertex.eval().scalar());
  }

  private void isGreaterThan(double a, double b, boolean expected) {
    BoolVertex vertex = ConstantVertex.of(a).greaterThan(ConstantVertex.of(b));
    assertEquals(expected, vertex.eval().scalar());
  }
}
