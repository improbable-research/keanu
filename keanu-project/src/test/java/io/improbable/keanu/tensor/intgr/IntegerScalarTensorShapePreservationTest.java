package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.Tensor;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.shape.Shape;

import static org.junit.Assert.assertArrayEquals;

public class IntegerScalarTensorShapePreservationTest {

    private static IntegerTensor intTensor;
    private static IntegerTensor scalarIntTensor;

    @Before
    public void setup() {
        intTensor = IntegerTensor.create(new int[]{5, 6, 7, 8, 9, 10}, new long[]{3, 2});
        scalarIntTensor = IntegerTensor.create(1, new long[]{1, 1, 1});
    }

    @Test
    public void integerScalarMultiplicationPreservesShape() {
        Tensor multiplicationResult = intTensor.times(scalarIntTensor);
        resultShapeMatchesBroadcastShape(multiplicationResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarAdditionPreservesShape() {
        Tensor additionResult = intTensor.plus(scalarIntTensor);
        resultShapeMatchesBroadcastShape(additionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarDivisionPreservesShape() {
        Tensor divisionResult = intTensor.div(scalarIntTensor);
        resultShapeMatchesBroadcastShape(divisionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarSubtractionPreservesShape() {
        Tensor subtractionResult = intTensor.minus(scalarIntTensor);
        resultShapeMatchesBroadcastShape(subtractionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarMultiplicationPreservesShapeReverse() {
        Tensor multiplicationResult = scalarIntTensor.times(intTensor);
        resultShapeMatchesBroadcastShape(multiplicationResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarAdditionPreservesShapeReverse() {
        Tensor additionResult = scalarIntTensor.plus(intTensor);
        resultShapeMatchesBroadcastShape(additionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarDivisionPreservesShapeReverse() {
        Tensor divisionResult = scalarIntTensor.div(intTensor);
        resultShapeMatchesBroadcastShape(divisionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarSubtractionPreservesShapeReverse() {
        Tensor subtractionResult = scalarIntTensor.minus(intTensor);
        resultShapeMatchesBroadcastShape(subtractionResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarPowerPreservesShape() {
        Tensor powerResult = intTensor.pow(scalarIntTensor);
        resultShapeMatchesBroadcastShape(powerResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarPowerPreservesShapeReverse() {
        Tensor powerResult = scalarIntTensor.pow(intTensor);
        resultShapeMatchesBroadcastShape(powerResult, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetGreaterThanMaskPreservesShape() {
        Tensor result = intTensor.getGreaterThanMask(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetGreaterThanMaskPreservesShapeReverse() {
        Tensor result = scalarIntTensor.getGreaterThanMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetGreaterThanOrEqualToMaskPreservesShape() {
        Tensor result = intTensor.getGreaterThanOrEqualToMask(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetGreaterThanOrEqualToMaskPreservesShapeReverse() {
        Tensor result = scalarIntTensor.getGreaterThanOrEqualToMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetLessThanMaskPreservesShape() {
        Tensor result = intTensor.getLessThanMask(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetLessThanMaskPreservesShapeReverse() {
        Tensor result = scalarIntTensor.getLessThanMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetLessThanOrEqualToMaskPreservesShape() {
        Tensor result = intTensor.getLessThanOrEqualToMask(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGetLessThanOrEqualToMaskPreservesShapeReverse() {
        Tensor result = scalarIntTensor.getLessThanOrEqualToMask(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarMaxInPlacePreservesShape() {
        Tensor result = intTensor.maxInPlace(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarMaxInPlacePreservesShapeReverse() {
        Tensor result = scalarIntTensor.maxInPlace(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarMinInPlacePreservesShape() {
        Tensor result = intTensor.minInPlace(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarMinInPlacePreservesShapeReverse() {
        Tensor result = scalarIntTensor.minInPlace(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarLessThanPreservesShape() {
        Tensor result = intTensor.lessThan(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarLessThanPreservesShapeReverse() {
        Tensor result = scalarIntTensor.lessThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarLessThanOrEqualPreservesShape() {
        Tensor result = intTensor.lessThanOrEqual(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarLessThanOrEqualPreservesShapeReverse() {
        Tensor result = scalarIntTensor.lessThanOrEqual(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGreaterThanPreservesShape() {
        Tensor result = intTensor.greaterThan(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGreaterThanPreservesShapeReverse() {
        Tensor result = scalarIntTensor.greaterThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGreaterThanOrEqualPreservesShape() {
        Tensor result = intTensor.greaterThanOrEqual(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarGreaterThanOrEqualPreservesShapeReverse() {
        Tensor result = scalarIntTensor.greaterThan(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarElementwiseEqualsPreservesShape() {
        Tensor result = intTensor.elementwiseEquals(scalarIntTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    @Test
    public void integerScalarElementwiseEqualsPreservesShapeReverse() {
        Tensor result = scalarIntTensor.elementwiseEquals(intTensor);
        resultShapeMatchesBroadcastShape(result, intTensor, scalarIntTensor);
    }

    private static void resultShapeMatchesBroadcastShape(Tensor result, Tensor input1, Tensor input2) {
        long[] broadcastShape = Shape.broadcastOutputShape(input1.getShape(), input2.getShape());
        long[] resultShape = result.getShape();
        assertArrayEquals(broadcastShape, resultShape);
    }
}
