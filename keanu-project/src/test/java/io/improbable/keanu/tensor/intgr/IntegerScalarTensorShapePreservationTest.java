package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.shape.Shape;

import java.util.function.BiFunction;

import static org.junit.Assert.assertArrayEquals;

public class IntegerScalarTensorShapePreservationTest {

    private static Nd4jIntegerTensor intTensor;
    private static Nd4jIntegerTensor lengthOneTensor;
    private static IntegerTensor scalarIntTensor;
    private static IntegerTensor[] tensors;


    @Before
    public void setup() {
        intTensor = new Nd4jIntegerTensor(new int[]{5, 6, 7, 8, 9, 10}, new long[]{3, 2});
        lengthOneTensor = new Nd4jIntegerTensor(new int[]{1}, new long[]{1, 1, 1});
        scalarIntTensor = IntegerTensor.scalar(1);
        tensors = new IntegerTensor[]{intTensor, lengthOneTensor, scalarIntTensor};
    }

    @Test
    public void tensorMultiplicationPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::times);
        checkOperationPreservesShape(IntegerTensor::timesInPlace);
    }

    @Test
    public void tensorAdditionPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::plus);
        checkOperationPreservesShape(IntegerTensor::plusInPlace);

    }

    @Test
    public void tensorDivisionPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::div);
        checkOperationPreservesShape(IntegerTensor::divInPlace);

    }

    @Test
    public void tensorSubtractionPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::minus);
        checkOperationPreservesShape(IntegerTensor::minusInPlace);
    }

    @Test
    public void tensorPowerPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::pow);
        checkOperationPreservesShape(IntegerTensor::powInPlace);

    }

    @Test
    public void tensorSetWithMaskInPlacePreservesShape() {
        for (IntegerTensor t1 : tensors) {
            IntegerTensor t2 = t1.duplicate();
            Tensor result = t1.setWithMaskInPlace(t2, -1);
            resultShapeMatchesBroadcastShape(result, t1, t2);
        }
    }

    @Test
    public void tensorGetGreaterThanMaskPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::greaterThanMask);

    }

    @Test
    public void tensorGetGreaterThanOrEqualToMaskPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::greaterThanOrEqualToMask);

    }

    @Test
    public void tensorGetLessThanMaskPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::lessThanMask);

    }

    @Test
    public void tensorGetLessThanOrEqualToMaskPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::lessThanOrEqualToMask);

    }

    @Test
    public void tensorMaxPreservesShape() {
        checkOperationPreservesShape(NumberTensor::max);
        checkOperationPreservesShape(NumberTensor::maxInPlace);

    }

    @Test
    public void tensorMinPreservesShape() {
        checkOperationPreservesShape(NumberTensor::min);
        checkOperationPreservesShape(NumberTensor::minInPlace);

    }

    @Test
    public void tensorLessThanPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::lessThan);
    }

    @Test
    public void tensorLessThanOrEqualPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::lessThanOrEqual);

    }

    @Test
    public void tensorGreaterThanPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::greaterThan);

    }

    @Test
    public void tensorGreaterThanOrEqualPreservesShape() {
        checkOperationPreservesShape(IntegerTensor::greaterThanOrEqual);

    }

    @Test
    public void tensorElementwiseEqualsPreservesShape() {
        checkOperationPreservesShape((l, r) -> l.elementwiseEquals(r));

    }

    private void checkOperationPreservesShape(BiFunction<IntegerTensor, IntegerTensor, Tensor> operation) {
        for (IntegerTensor t1 : tensors) {
            for (IntegerTensor t2 : tensors) {
                Tensor result = operation.apply(t1, t2);
                resultShapeMatchesBroadcastShape(result, t1, t2);
            }
        }
    }

    private static void resultShapeMatchesBroadcastShape(Tensor result, Tensor input1, Tensor input2) {
        long[] broadcastShape = Shape.broadcastOutputShape(input1.getShape(), input2.getShape());
        long[] resultShape = result.getShape();
        assertArrayEquals(broadcastShape, resultShape);
    }
}
