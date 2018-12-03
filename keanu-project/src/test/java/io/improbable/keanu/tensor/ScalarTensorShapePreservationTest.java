package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.shape.Shape;

import static org.junit.Assert.assertArrayEquals;

public class ScalarTensorShapePreservationTest {

    private static DoubleTensor doubleTensor;
    private static DoubleTensor scalarDoubleTensor;
    private static IntegerTensor intTensor;
    private static IntegerTensor scalarIntTensor;

    @Before
    public void setup() {
        doubleTensor = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, new long[]{2, 3});
        scalarDoubleTensor = DoubleTensor.create(1.0, new long[]{1, 1, 1, 1});

        intTensor = IntegerTensor.create(new int[]{5, 6, 7, 8, 9, 10}, new long[]{3, 2});
        scalarIntTensor = IntegerTensor.create(1, new long[]{1, 1, 1});
    }

    @Test
    public void doubleScalarMultiplicationPreservesShape() {
        Tensor multiplicationResult = doubleTensor.times(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(multiplicationResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarAdditionPreservesShape() {
        Tensor additionResult = doubleTensor.plus(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(additionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarDivisionPreservesShape() {
        Tensor divisionResult = doubleTensor.div(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(divisionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarSubtractionPreservesShape() {
        Tensor subtractionResult = doubleTensor.minus(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(subtractionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarMultiplicationPreservesShape() {
        Tensor multiplicationResult = scalarIntTensor.times(intTensor);
        resultShapeMatchesBroadcastShape(multiplicationResult, intTensor, scalarIntTensor);
    }

    @Test
    public void intScalarAdditionPreservesShape() {
        Tensor additionResult = scalarIntTensor.plus(intTensor);
        resultShapeMatchesBroadcastShape(additionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void intScalarDivisionPreservesShape() {
        Tensor divisionResult = scalarIntTensor.div(intTensor);
        resultShapeMatchesBroadcastShape(divisionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void intScalarSubtractionPreservesShape() {
        Tensor subtractionResult = scalarIntTensor.minus(intTensor);
        resultShapeMatchesBroadcastShape(subtractionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarPowerPreservesShape() {
        Tensor powerResult = doubleTensor.pow(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(powerResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarPowerPreservesShape() {
        Tensor powerResult = scalarIntTensor.pow(intTensor);
        resultShapeMatchesBroadcastShape(powerResult, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarAtan2PreservesShape() {
        Tensor atan2Result = doubleTensor.atan2(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(atan2Result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarAtan2PreservesShapeReverse() {
        Tensor atan2Result = scalarDoubleTensor.atan2(doubleTensor);
        resultShapeMatchesBroadcastShape(atan2Result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetGreaterThanMaskPreservesShape() {
        Tensor result = doubleTensor.getGreaterThanMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGetGreaterThanMaskPreservesShape() {
        Tensor result = scalarIntTensor.getGreaterThanMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarGetGreaterThanOrEqualToMaskPreservesShape() {
        Tensor result = doubleTensor.getGreaterThanOrEqualToMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGetGreaterThanOrEqualToMaskPreservesShape() {
        Tensor result = scalarIntTensor.getGreaterThanOrEqualToMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarGetLessThanMaskPreservesShape() {
        Tensor result = doubleTensor.getLessThanMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGetLessThanMaskPreservesShape() {
        Tensor result = scalarIntTensor.getLessThanMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarGetLessThanOrEqualToMaskPreservesShape() {
        Tensor result = doubleTensor.getLessThanOrEqualToMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGetLessThanOrEqualToMaskPreservesShape() {
        Tensor result = scalarIntTensor.getLessThanOrEqualToMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarMaxInPlacePreservesShape() {
        Tensor result = doubleTensor.maxInPlace(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarMaxInPlacePreservesShape() {
        Tensor result = scalarIntTensor.maxInPlace(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarMinInPlacePreservesShape() {
        Tensor result = doubleTensor.minInPlace(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarMinInPlacePreservesShape() {
        Tensor result = scalarIntTensor.minInPlace(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarLessThanPreservesShape() {
        Tensor result = doubleTensor.lessThan(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarLessThanPreservesShape() {
        Tensor result = scalarIntTensor.lessThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarLessThanOrEqualPreservesShape() {
        Tensor result = doubleTensor.lessThanOrEqual(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarLessThanOrEqualPreservesShape() {
        Tensor result = scalarIntTensor.lessThanOrEqual(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarGreaterThanPreservesShape() {
        Tensor result = doubleTensor.greaterThan(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGreaterThanPreservesShape() {
        Tensor result = scalarIntTensor.greaterThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarGreaterThanOrEqualPreservesShape() {
        Tensor result = doubleTensor.greaterThanOrEqual(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarGreaterThanOrEqualPreservesShape() {
        Tensor result = scalarIntTensor.greaterThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void doubleScalarElementwiseEqualsPreservesShape() {
        Tensor result = doubleTensor.elementwiseEquals(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void intScalarElementwiseEqualsPreservesShape() {
        Tensor result = scalarIntTensor.elementwiseEquals(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    private static void resultShapeMatchesBroadcastShape(Tensor result, Tensor input1, Tensor input2) {
        long[] broadcastShape = Shape.broadcastOutputShape(input1.getShape(), input2.getShape());
        long[] resultShape = result.getShape();
        assertArrayEquals(broadcastShape, resultShape);
    }
}
