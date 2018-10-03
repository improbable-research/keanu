package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAScalarAndVector;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAVectorAndScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoMatricesElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoScalars;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

public class PowerVertexTest {

  @Test
  public void powerTwoScalarVertexValues() {
    operatesOnTwoScalarVertexValues(2.0, 3.0, 8.0, DoubleVertex::pow);
  }

  @Test
  public void calculatesDualNumberOfTwoScalarsPower() {
    calculatesDualNumberOfTwoScalars(2.0, 3.0, 3. * 4., Math.log(2.) * 8., DoubleVertex::pow);
  }

  @Test
  public void powerTwoMatrixVertexValues() {
    operatesOnTwo2x2MatrixVertexValues(
        new double[] {1.0, 2.0, 3.0, 4.0},
        new double[] {2.0, 3.0, 4.0, 5.0},
        new double[] {1.0, 8.0, 81.0, 1024.0},
        DoubleVertex::pow);
  }

  @Test
  public void calculatesDualNumberOfTwoMatricesElementWisePower() {
    calculatesDualNumberOfTwoMatricesElementWiseOperator(
        DoubleTensor.create(new double[] {1.0, 2.0, 3.0, 4.0}, 1, 4),
        DoubleTensor.create(new double[] {2.0, 3.0, 4.0, 5.0}, 1, 4),
        DoubleTensor.create(new double[] {2.0, 3.0 * 4, 4.0 * 27, 5.0 * 256})
            .diag()
            .reshape(1, 4, 1, 4),
        DoubleTensor.create(
                new double[] {
                  Math.log(1.0) * 1, Math.log(2.0) * 8, Math.log(3.0) * 81, Math.log(4.0) * 1024
                })
            .diag()
            .reshape(1, 4, 1, 4),
        DoubleVertex::pow);
  }

  @Test
  public void calculatesDualNumberOfAVectorsAndScalarPower() {
    calculatesDualNumberOfAVectorAndScalar(
        DoubleTensor.create(new double[] {1.0, 2.0, 3.0, 4.0}),
        3,
        DoubleTensor.create(new double[] {3.0, 3.0 * 4., 3.0 * 9, 3.0 * 16})
            .diag()
            .reshape(1, 4, 1, 4),
        DoubleTensor.create(
                new double[] {
                  Math.log(1.0), Math.log(2.0) * 8, Math.log(3.0) * 27, Math.log(4.0) * 64
                })
            .reshape(1, 4, 1, 1),
        DoubleVertex::pow);
  }

  @Test
  public void calculatesDualNumberofAScalarAndVectorPower() {
    calculatesDualNumberOfAScalarAndVector(
        3,
        DoubleTensor.create(new double[] {1.0, 2.0, 3.0, 4.0}),
        DoubleTensor.create(new double[] {1., 2.0 * 3, 3.0 * 9., 4.0 * 27}).reshape(1, 4, 1, 1),
        DoubleTensor.create(
                new double[] {
                  Math.log(3.0) * 3, Math.log(3.0) * 9, Math.log(3.0) * 27, Math.log(3.0) * 81
                })
            .diag()
            .reshape(1, 4, 1, 4),
        DoubleVertex::pow);
  }
}
