package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

public class MaxVertex extends DoubleIfVertex {

  /**
   * Finds the maximum between two vertices
   *
   * @param left one of the vertices to find the maximum of
   * @param right one of the vertices to find the maximum of
   */
  public MaxVertex(DoubleVertex left, DoubleVertex right) {
    super(left.getShape(), left.greaterThanOrEqualTo(right), left, right);
  }
}
