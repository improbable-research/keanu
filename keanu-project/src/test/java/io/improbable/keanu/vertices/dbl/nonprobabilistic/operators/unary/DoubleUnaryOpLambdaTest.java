package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertArrayEquals;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

public class DoubleUnaryOpLambdaTest {

  @Test
  public void GIVEN_a_double_tensor_THEN_transform() {

    UniformVertex matrix = new UniformVertex(new int[] {2, 2}, 0, 5);
    matrix.setAndCascade(DoubleTensor.create(2.5, new int[] {2, 2}));
    DoubleVertex matrixLambda = matrix.lambda((val) -> val.times(2), null, null);

    assertArrayEquals(
        new double[] {5, 5, 5, 5}, matrixLambda.getValue().asFlatDoubleArray(), 0.001);
  }
}
