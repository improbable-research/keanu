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

public class ExpVertexTest {

  @Test
  public void expScalarVertexValue() {
    operatesOnScalarVertexValue(5, Math.exp(5), DoubleVertex::exp);
  }

  @Test
  public void calculatesDualNumberOScalarExp() {
    calculatesDualNumberOfScalar(0.5, Math.exp(0.5), DoubleVertex::exp);
  }

  @Test
  public void expMatrixVertexValues() {
    operatesOn2x2MatrixVertexValues(
        new double[] {0.0, 0.1, 0.2, 0.3},
        new double[] {Math.exp(0.0), Math.exp(0.1), Math.exp(0.2), Math.exp(0.3)},
        DoubleVertex::exp);
  }

  @Test
  public void calculatesDualNumberOfMatrixElementWiseexp() {
    calculatesDualNumberOfMatrixElementWiseOperator(
        new double[] {0.1, 0.2, 0.3, 0.4},
        toDiagonalArray(new double[] {Math.exp(0.1), Math.exp(0.2), Math.exp(0.3), Math.exp(0.4)}),
        DoubleVertex::exp);
  }

  @Test
  public void changesMatchGradient() {
    DoubleVertex inputVertex = new UniformVertex(new int[] {2, 2, 2}, -10.0, 10.0);
    DoubleVertex outputVertex = inputVertex.div(3).exp();

    finiteDifferenceMatchesGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-5, true);
  }
}
