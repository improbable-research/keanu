package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.*;

public class ArcTan2VertexTest {

    @Test
    public void tan2TwoScalarVertexValues() {
        operatesOnTwoScalarVertexValues(
            1.0,
            Math.PI,
            Math.atan2(Math.PI, 1.0),
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDualNumberOfTwoScalarsTan2() {
        double a = 0.5;
        double b = Math.sqrt(3) / 2.0;
        double wrtA = b / (Math.pow(b, 2) * Math.pow(0.5, 2));
        double wrtB = -0.5 / (Math.pow(b, 2) * Math.pow(0.5, 2));

        calculatesDualNumberOfTwoScalars(
            a,
            b,
            wrtA,
            wrtB,
            DoubleVertex::atan2
        );
    }

    @Test
    public void tan2MatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{0.0, 0.1, 0.2, 0.3},
            new double[]{2.0, 4.0, 3.0, 8.0},
            new double[]{Math.atan2(2., 0.), Math.atan2(4, .1), Math.atan2(3, .2), Math.atan2(8, .3)},
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDualNumberOfTwoMatricesElementWiseTan2() {
        calculatesDualNumberOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 1, 4),
            DoubleTensor.create(new double[]{2. / 4., 3. / 36., 4. / (9. * 16), 5. / (16 * 25)}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{-1. / 4., -2. / 36., -3. / (9 * 16), -4. / (16 * 25)}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::atan2
        );
    }

}
