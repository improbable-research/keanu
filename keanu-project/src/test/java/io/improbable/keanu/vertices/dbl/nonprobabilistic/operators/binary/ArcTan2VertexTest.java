package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAScalarAndVector;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfAVectorAndScalar;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoMatricesElementWiseOperator;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.calculatesDualNumberOfTwoScalars;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

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
        double x = 0.5;
        double y = Math.sqrt(3) / 2.0;
        double wrtX = -y / (Math.pow(y, 2) + Math.pow(x, 2));
        double wrtY = x / (Math.pow(y, 2) + Math.pow(x, 2));

        calculatesDualNumberOfTwoScalars(
            x,
            y,
            wrtX,
            wrtY,
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
            DoubleTensor.create(new double[]{-2. / (1 + 4), -3. / (4 + 9), -4. / (9. + 16), -5. / (16 + 25)}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1. / (1 + 4), 2. / (4 + 9), 3. / (9 + 16), 4. / (16 + 25)}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDualNumberOfAVectorsAndScalarTan2() {
        calculatesDualNumberOfAVectorAndScalar(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            2,
            DoubleTensor.create(new double[]{-2. / (1. + 4.), -2. / (4. + 4.), -2. / (9. + 4.), -2. / (16. + 4.)}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1. / (1. + 4.), 2. / (4. + 4.), 3. / (9. + 4.), 4. / (16. + 4.)}).reshape(1, 4, 1, 1),
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDualNumberofAScalarAndVectorsTan2() {
        calculatesDualNumberOfAScalarAndVector(
            2,
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}),
            DoubleTensor.create(new double[]{-1. / (4. + 1.), -2. / (4. + 4.), -3. / (4. + 9.), -4. / (4. + 16.)}).reshape(1, 4, 1, 1),
            DoubleTensor.create(new double[]{2. / (4. + 1.), 2. / (4. + 4.), 2. / (4. + 9.), 2. / (4. + 16.)}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::atan2
        );
    }

}
