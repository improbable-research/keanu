package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.calculatesDerivativeOfAScalarAndVector;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.calculatesDerivativeOfAVectorAndScalar;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.calculatesDerivativeOfTwoMatricesElementWiseOperator;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.calculatesDerivativeOfTwoScalars;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

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
    public void calculatesDerivativeOfTwoScalarsTan2() {
        double x = 0.5;
        double y = Math.sqrt(3) / 2.0;
        double wrtX = -y / (Math.pow(y, 2) + Math.pow(x, 2));
        double wrtY = x / (Math.pow(y, 2) + Math.pow(x, 2));

        calculatesDerivativeOfTwoScalars(
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
    public void calculatesDerivativeOfTwoMatricesElementWiseTan2() {
        calculatesDerivativeOfTwoMatricesElementWiseOperator(
            DoubleTensor.create(new double[]{1.0, 2.0, 3.0, 4.0}, 1, 4),
            DoubleTensor.create(new double[]{2.0, 3.0, 4.0, 5.0}, 1, 4),
            DoubleTensor.create(new double[]{-2. / (1 + 4), -3. / (4 + 9), -4. / (9. + 16), -5. / (16 + 25)}).diag().reshape(1, 4, 1, 4),
            DoubleTensor.create(new double[]{1. / (1 + 4), 2. / (4 + 9), 3. / (9 + 16), 4. / (16 + 25)}).diag().reshape(1, 4, 1, 4),
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDerivativeOfAVectorsAndScalarTan2() {
        calculatesDerivativeOfAVectorAndScalar(
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            2,
            DoubleTensor.create(new double[]{-2. / (1. + 4.), -2. / (4. + 4.), -2. / (9. + 4.), -2. / (16. + 4.)}).diag().reshape(4, 4),
            DoubleTensor.create(new double[]{1. / (1. + 4.), 2. / (4. + 4.), 3. / (9. + 4.), 4. / (16. + 4.)}).reshape(4),
            DoubleVertex::atan2
        );
    }

    @Test
    public void calculatesDerivativeofAScalarAndVectorsTan2() {
        calculatesDerivativeOfAScalarAndVector(
            2,
            DoubleTensor.create(1.0, 2.0, 3.0, 4.0),
            DoubleTensor.create(new double[]{-1. / (4. + 1.), -2. / (4. + 4.), -3. / (4. + 9.), -4. / (4. + 16.)}).reshape(4),
            DoubleTensor.create(new double[]{2. / (4. + 1.), 2. / (4. + 4.), 2. / (4. + 9.), 2. / (4. + 16.)}).diag().reshape(4, 4),
            DoubleVertex::atan2
        );
    }

}
