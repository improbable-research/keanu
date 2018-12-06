package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.shape.Shape;

import static org.junit.Assert.assertArrayEquals;

public class DoubleScalarTensorShapePreservationTest {

    private static DoubleTensor doubleTensor;
    private static DoubleTensor scalarDoubleTensor;

    @Before
    public void setup() {
        doubleTensor = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, new long[]{2, 3});
        scalarDoubleTensor = DoubleTensor.create(1.0, new long[]{1, 1, 1, 1});
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
    public void doubleScalarMultiplicationPreservesShapeReverse() {
        Tensor multiplicationResult = scalarDoubleTensor.times(doubleTensor);
        resultShapeMatchesBroadcastShape(multiplicationResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarAdditionPreservesShapeReverse() {
        Tensor additionResult = scalarDoubleTensor.plus(doubleTensor);
        resultShapeMatchesBroadcastShape(additionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarDivisionPreservesShapeReverse() {
        Tensor divisionResult = scalarDoubleTensor.div(doubleTensor);
        resultShapeMatchesBroadcastShape(divisionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarSubtractionPreservesShapeReverse() {
        Tensor subtractionResult = scalarDoubleTensor.minus(doubleTensor);
        resultShapeMatchesBroadcastShape(subtractionResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarPowerPreservesShape() {
        Tensor powerResult = doubleTensor.pow(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(powerResult, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarPowerPreservesShapeReverse() {
        Tensor powerResult = scalarDoubleTensor.pow(doubleTensor);
        resultShapeMatchesBroadcastShape(powerResult, doubleTensor, scalarDoubleTensor);
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
    public void doubleScalarGetGreaterThanMaskPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.getGreaterThanMask(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetGreaterThanOrEqualToMaskPreservesShape() {
        Tensor result = doubleTensor.getGreaterThanOrEqualToMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetGreaterThanOrEqualToMaskPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.getGreaterThanOrEqualToMask(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetLessThanMaskPreservesShape() {
        Tensor result = doubleTensor.getLessThanMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetLessThanMaskPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.getLessThanMask(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetLessThanOrEqualToMaskPreservesShape() {
        Tensor result = doubleTensor.getLessThanOrEqualToMask(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGetLessThanOrEqualToMaskPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.getLessThanOrEqualToMask(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarMaxInPlacePreservesShape() {
        Tensor result = doubleTensor.maxInPlace(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarMaxInPlacePreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.maxInPlace(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarMinInPlacePreservesShape() {
        Tensor result = doubleTensor.minInPlace(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarMinInPlacePreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.minInPlace(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarLessThanPreservesShape() {
        Tensor result = doubleTensor.lessThan(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarLessThanPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.lessThan(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarLessThanOrEqualPreservesShape() {
        Tensor result = doubleTensor.lessThanOrEqual(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarLessThanOrEqualPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.lessThanOrEqual(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGreaterThanPreservesShape() {
        Tensor result = doubleTensor.greaterThan(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGreaterThanPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.greaterThan(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGreaterThanOrEqualPreservesShape() {
        Tensor result = doubleTensor.greaterThanOrEqual(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarGreaterThanOrEqualPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.greaterThan(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarElementwiseEqualsPreservesShape() {
        Tensor result = doubleTensor.elementwiseEquals(scalarDoubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    @Test
    public void doubleScalarElementwiseEqualsPreservesShapeReverse() {
        Tensor result = scalarDoubleTensor.elementwiseEquals(doubleTensor);
        resultShapeMatchesBroadcastShape(result, doubleTensor, scalarDoubleTensor);
    }

    private static void resultShapeMatchesBroadcastShape(Tensor result, Tensor input1, Tensor input2) {
        long[] broadcastShape = Shape.broadcastOutputShape(input1.getShape(), input2.getShape());
        long[] resultShape = result.getShape();
        assertArrayEquals(broadcastShape, resultShape);
    }
}
