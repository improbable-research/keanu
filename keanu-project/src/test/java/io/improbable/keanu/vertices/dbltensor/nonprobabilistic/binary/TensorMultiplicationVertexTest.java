package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary;

import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbltensor.nonprobabilistic.binary.BinaryOperationHelper.*;

public class TensorMultiplicationVertexTest {

    @Test
    public void multipliesTwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 6.0, DoubleTensorVertex::multiply);
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsMultiplied() {
        calculatesDualNumberOfTwoScalars(2.0, 3.0, 3.0, 2.0, DoubleTensorVertex::multiply);
    }

    @Test
    public void multipliesTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new double[]{2.0, 3.0, 4.0, 5.0},
                new double[]{2.0, 6.0, 12.0, 20.0},
                DoubleTensorVertex::multiply
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseMultiplied() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new double[]{2.0, 3.0, 4.0, 5.0},
                new double[]{2.0, 3.0, 4.0, 5.0},
                new double[]{1.0, 2.0, 3.0, 4.0},
                DoubleTensorVertex::multiply
        );
    }
}
