package io.improbable.keanu.tensor.dbl;

import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertDivideInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertDivideOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertMinusInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertMinusOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPlusInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPlusOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesOperationEquals;

import org.junit.Before;
import org.junit.Test;

public class Nd4jBroadcastTensorTest {

    Nd4jDoubleTensor matrixA;

    @Before
    public void setup() {
        matrixA = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
    }

    @Test
    public void canBroadcastMultiplyRank4ContainingVectorAndMatrix() {

        /*
          a = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2,2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2,1,1)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2, 1});
        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{1, 2, 6, 8, 15, 18, 28, 32}, new int[]{2, 2, 2, 1});

        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{1, 2, 6, 8, 3, 6, 12, 16}, new int[]{2, 2, 2});

        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyToHigherRankWithDimensionsOfOnePrepending() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(1,2,2)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{1,  4,  9, 16}, new int[]{1, 2, 2});

        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyRank4ContainingMatrixAndMatrix() {

        /*
          a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6]).reshape(2,2,2,2)
          b = np.array([1, 2, 3, 4]).reshape(2,2,1,1)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8,
            4, 3, 2, 1, 7, 5, 8, 6
        }, new int[]{2, 2, 2, 2});

        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 10, 12, 14, 16,
            12, 9, 6, 3, 28, 20, 32, 24
        }, new int[]{2, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyRank5ContainingMatrixAndMatrix() {

        /*
          a = np.array([
          1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
          6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
          ]).reshape(2,2,2,2,2)
          b = np.array([1, 2, 3, 4]).reshape(2,2,1,1,1)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank5 = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
        }, new int[]{2, 2, 2, 2, 2});

        DoubleTensor matrix = matrixA.reshape(2, 2, 1, 1, 1);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 8, 6, 4, 2, 14, 10, 16, 12,
            18, 9, 6, 27, 9, 12, 21, 18, 24, 8, 20, 16, 0, 8, 4, 12
        }, new int[]{2, 2, 2, 2, 2});

        assertTimesOperationEquals(rank5, matrix, expected);
        assertTimesInPlaceOperationEquals(rank5, matrix, expected);
    }

    @Test
    public void canSuperBroadcast() {

        /*
          a = np.zeros([2,2,2,2])
          b = np.array([1,0,1,0]).reshape(2,2)
          ab = a + b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor x = Nd4jDoubleTensor.zeros(new int[]{2, 2, 2, 2});
        DoubleTensor y = Nd4jDoubleTensor.create(new double[]{1, 0, 1, 0}, new int[]{2, 2});

        DoubleTensor diff = x.plus(y);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new int[]{2, 2, 2, 2});

        assertEquals(expected, diff);
    }

    @Test
    public void canSuperBroadcastInPlace() {
        DoubleTensor x = Nd4jDoubleTensor.zeros(new int[]{2, 2, 2, 2});
        DoubleTensor y = Nd4jDoubleTensor.create(new double[]{1, 0, 1, 0}, new int[]{2, 2});

        DoubleTensor diff = x.plusInPlace(y);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new int[]{2, 2, 2, 2});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastAdd() {

        /*
          x = np.array([1,2,3]).reshape(3,1)
          s = np.array([-5, -2, -3, -7, -8, -5, -2, -3, -7, -8, -5, -2, -3, -7, -8]).reshape(3,5)
          sx = s + x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        DoubleTensor diff = s.plus(x);
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastAddToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a + b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{2, 3, 5, 6, 4, 5, 7, 8}, new int[]{2, 2, 2});

        assertPlusOperationEquals(rank4, matrix, expected);
        assertPlusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastAddToHigherRankWithDimensionsOfOnePrepending() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(1,2,2)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a + b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{2, 4, 6, 8}, new int[]{1, 2, 2});

        assertPlusOperationEquals(rank4, matrix, expected);
        assertPlusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastSubtract() {

        /*
          x = np.array([-1,-2,-3]).reshape(3,1)
          s = np.array([-5, -2, -3, -7, -8, -5, -2, -3, -7, -8, -5, -2, -3, -7, -8]).reshape(3,5)
          sx = s - x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{-1, -2, -3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new int[]{3, 5});

        DoubleTensor diff = s.minus(x);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastSubtractToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a - b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{0, -1, -1, -2, 2, 1, 1, 0}, new int[]{2, 2, 2});

        assertMinusOperationEquals(rank4, matrix, expected);
        assertMinusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastSubtractToHigherRankWithDimensionsOfOnePrepending() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(1,2,2)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a - b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{0, 0, 0, 0}, new int[]{1, 2, 2});

        assertMinusOperationEquals(rank4, matrix, expected);
        assertMinusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastDivide() {

        /*
          x = np.array([1,2,3]).reshape(3,1)
          s = np.array([5, 2, 3, 7, 8, 5, 2, 3, 7, 8, 5, 2, 3, 7, 8]).reshape(3,5)
          sx = s / x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = Nd4jDoubleTensor.create(new double[]{1, 2, 3}, new int[]{3, 1});
        DoubleTensor s = Nd4jDoubleTensor.create(new double[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new int[]{3, 5});

        DoubleTensor diff = s.div(x);

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            5 / 1.0, 2 / 1.0, 3 / 1.0, 7 / 1.0, 8 / 1.0,
            5 / 2.0, 2 / 2.0, 3 / 2.0, 7 / 2.0, 8 / 2.0,
            5 / 3.0, 2 / 3.0, 3 / 3.0, 7 / 3.0, 8 / 3.0
        }, new int[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivideToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([10, 20, 30, 40]).reshape(2,2,1)
          b = np.array([1, 2, 5, 10]).reshape(2,2)
          ab = a / b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{10, 20, 30, 40}, new int[]{2, 2, 1});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 5, 10}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{10., 5., 4., 2., 30., 15., 8., 4.}, new int[]{2, 2, 2});

        assertDivideOperationEquals(rank4, matrix, expected);
        assertDivideInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastDivideToHigherRankWithDimensionsOfOnePrepending() {

        /*
          a = np.array([10, 20, 30, 40]).reshape(1,2,2)
          b = np.array([1, 2, 5, 10]).reshape(2,2)
          ab = a / b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2});
        DoubleTensor matrix = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{1., 1., 1., 1.}, new int[]{1, 2, 2});

        assertDivideOperationEquals(rank4, matrix, expected);
        assertDivideInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});


        assertTimesOperationEquals(matrix, rank4, expected);
        assertTimesInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.zeros(new int[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});

        assertPlusOperationEquals(rank4, matrix, expected);
        assertPlusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.zeros(new int[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new int[]{4, 2, 2, 2});

        assertPlusOperationEquals(matrix, rank4, expected);
        assertPlusInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
        }, new int[]{4, 2, 2, 2});

        assertDivideOperationEquals(rank4, matrix, expected);
        assertDivideInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
        }, new int[]{4, 2, 2, 2});

        assertDivideOperationEquals(matrix, rank4, expected);
        assertDivideInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1
        }, new int[]{4, 2, 2, 2});

        assertMinusOperationEquals(rank4, matrix, expected);
        assertMinusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new int[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor expected = Nd4jDoubleTensor.create(new double[]{
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1
        }, new int[]{4, 2, 2, 2});

        assertMinusOperationEquals(matrix, rank4, expected);
        assertMinusInPlaceOperationEquals(matrix, rank4, expected);
    }

}
