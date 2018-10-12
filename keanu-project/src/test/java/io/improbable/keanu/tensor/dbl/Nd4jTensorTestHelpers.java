package io.improbable.keanu.tensor.dbl;

import static org.junit.Assert.assertEquals;

public class Nd4jTensorTestHelpers {

    public static void assertTimesOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.times(right);
        assertEquals(expected, actual);
    }

    public static void assertTimesInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.timesInPlace(right);
        assertEquals(expected, left);
    }

    public static void assertPlusOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.plus(right);
        assertEquals(expected, actual);
    }

    public static void assertPlusInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.plusInPlace(right);
        assertEquals(expected, left);
    }

    public static void assertDivideOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.div(right);
        assertEquals(expected, actual);
    }

    public static void assertDivideInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.divInPlace(right);
        assertEquals(expected, left);
    }

    public static void assertMinusOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        DoubleTensor actual = left.minus(right);
        assertEquals(expected, actual);
    }

    public static void assertMinusInPlaceOperationEquals(DoubleTensor left, DoubleTensor right, DoubleTensor expected) {
        left = left.minusInPlace(right);
        assertEquals(expected, left);
    }

}
