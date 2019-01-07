package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.shape.Shape;

import java.util.function.BiFunction;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;

public class DoubleScalarTensorShapePreservationTest {

    private static Nd4jDoubleTensor doubleTensor;
    private static Nd4jDoubleTensor lengthOneTensor;
    private static ScalarDoubleTensor scalarDoubleTensor;
    private static DoubleTensor[] tensors;

    @Before
    public void setup() {
        doubleTensor = new Nd4jDoubleTensor(new double[]{1, 2, 3, 4, 5, 6}, new long[]{2, 3});
        lengthOneTensor = new Nd4jDoubleTensor(new double[]{1.0}, new long[]{1, 1, 1, 1});
        scalarDoubleTensor = new ScalarDoubleTensor(1.);
        tensors = new DoubleTensor[]{doubleTensor, lengthOneTensor, scalarDoubleTensor};
    }

    @Test
    public void tensorMultiplicationPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::times);
        checkOperationPreservesShape(DoubleTensor::timesInPlace);
    }

    @Test
    public void tensorAdditionPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::plus);
        checkOperationPreservesShape(DoubleTensor::plusInPlace);

    }

    @Test
    public void tensorDivisionPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::div);
        checkOperationPreservesShape(DoubleTensor::divInPlace);

    }

    @Test
    public void tensorSubtractionPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::minus);
        checkOperationPreservesShape(DoubleTensor::minusInPlace);

    }

    @Test
    public void tensorPowerPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::pow);
        checkOperationPreservesShape(DoubleTensor::powInPlace);
    }

    @Test
    public void tensorAtan2PreservesShape() {
        checkOperationPreservesShape(DoubleTensor::atan2);
        checkOperationPreservesShape(DoubleTensor::atan2InPlace);

    }

    @Test
    public void tensorSetWithMaskInPlacePreservesShape() {
        for (DoubleTensor t1 : tensors) {
            DoubleTensor t2 = t1.duplicate();
            Tensor result = t1.setWithMaskInPlace(t2, -1.);
            resultShapeMatchesBroadcastShape(result, t1, t2);
        }
    }

    @Test
    public void tensorGetGreaterThanMaskPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::getGreaterThanMask);
    }

    @Test
    public void tensorGetGreaterThanOrEqualToMaskPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::getGreaterThanOrEqualToMask);

    }

    @Test
    public void tensorGetLessThanMaskPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::getLessThanMask);

    }

    @Test
    public void tensorGetLessThanOrEqualToMaskPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::getLessThanOrEqualToMask);

    }

    @Test
    public void tensorMaxPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::max);
        checkOperationPreservesShape(DoubleTensor::maxInPlace);

    }

    @Test
    public void tensorMinPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::min);
        checkOperationPreservesShape(DoubleTensor::minInPlace);

    }

    @Test
    public void tensorLessThanPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::lessThan);

    }

    @Test
    public void tensorLessThanOrEqualPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::lessThanOrEqual);

    }

    @Test
    public void tensorGreaterThanPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::greaterThan);

    }

    @Test
    public void tensorGreaterThanOrEqualPreservesShape() {
        checkOperationPreservesShape(DoubleTensor::greaterThanOrEqual);

    }

    @Test
    public void tensorElementwiseEqualsPreservesShape() {
        checkOperationPreservesShape((l,r) -> l.elementwiseEquals(r));
    }


    private void checkOperationPreservesShape(BiFunction<DoubleTensor, DoubleTensor, Tensor> operation) {
        for (DoubleTensor t1 : tensors) {
            for (DoubleTensor t2 : tensors) {
                Tensor result = operation.apply(t1, t2);
                resultShapeMatchesBroadcastShape(result, t1, t2);
            }
        }
    }

    private static void resultShapeMatchesBroadcastShape(Tensor result, Tensor input1, Tensor input2) {
        long[] broadcastShape = Shape.broadcastOutputShape(input1.getShape(), input2.getShape());
        long[] resultShape = result.getShape();
        assertThat(broadcastShape, equalTo(resultShape));
    }
}
