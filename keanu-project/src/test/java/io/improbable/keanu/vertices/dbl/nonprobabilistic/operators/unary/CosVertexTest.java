package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.toDiagonalArray;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfMatrixElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.calculatesDualNumberOfScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOn2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.UnaryOperationTestHelpers.operatesOnScalarVertexValue;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

public class CosVertexTest {

  @Test
  public void cosScalarVertexValue() {
    operatesOnScalarVertexValue(Math.PI, Math.cos(Math.PI), DoubleVertex::cos);
  }

  @Test
  public void calculatesDualNumberOScalarCos() {
    calculatesDualNumberOfScalar(0.5, -Math.sin(0.5), DoubleVertex::cos);
  }

  @Test
  public void cosMatrixVertexValues() {
    operatesOn2x2MatrixVertexValues(
        new double[] {0.0, 0.1, 0.2, 0.3},
        new double[] {Math.cos(0.0), Math.cos(0.1), Math.cos(0.2), Math.cos(0.3)},
        DoubleVertex::cos);
  }

  @Test
  public void calculatesDualNumberOfMatrixElementWiseCos() {
    calculatesDualNumberOfMatrixElementWiseOperator(
        new double[] {0.1, 0.2, 0.3, 0.4},
        toDiagonalArray(
            new double[] {-Math.sin(0.1), -Math.sin(0.2), -Math.sin(0.3), -Math.sin(0.4)}),
        DoubleVertex::cos);
  }

  @Test
  public void changesMatchGradient() {
    DoubleVertex inputVertex = new UniformVertex(new int[] {2, 2, 2}, -10.0, 10.0);
    DoubleVertex outputVertex = inputVertex.times(3).cos();

    finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-5, true);
  }
}
