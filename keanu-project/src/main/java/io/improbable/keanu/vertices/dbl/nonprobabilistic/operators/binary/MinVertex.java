package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

public class MinVertex extends DoubleIfVertex {

  /**
   * Finds the minimum between two vertices
   *
   * @param left one of the vertices to find the minimum of
   * @param right one of the vertices to find the minimum of
   */
  public MinVertex(DoubleVertex left, DoubleVertex right) {
    super(left.getShape(), left.lessThanOrEqualTo(right), left, right);
  }
}
