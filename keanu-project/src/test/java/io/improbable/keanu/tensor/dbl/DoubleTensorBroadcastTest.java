package io.improbable.keanu.tensor.dbl;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;

import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertDivideInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertDivideOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertMinusInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertMinusOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPlusInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPlusOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPowInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertPowOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesInPlaceOperationEquals;
import static io.improbable.keanu.tensor.dbl.Nd4jTensorTestHelpers.assertTimesOperationEquals;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class DoubleTensorBroadcastTest {

    @Parameterized.Parameters(name = "{index}: Test with {1}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {new Nd4jDoubleTensorFactory(), "ND4J DoubleTensor"},
            {new JVMDoubleTensorFactory(), "JVM DoubleTensor"},
        });
    }

    public DoubleTensorBroadcastTest(DoubleTensorFactory factory, String name) {
        DoubleTensor.setFactory(factory);
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

        DoubleTensor rank4 = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2, 1});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2}).reshape(2, 2, 1, 1);
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 2, 6, 8, 15, 18, 28, 32}, new long[]{2, 2, 2, 1});

        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test(expected = IllegalArgumentException.class)
    public void canBroadcastMultiplyToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2, 1});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 2, 6, 8, 3, 6, 12, 16}, new long[]{2, 2, 2});

        assertTimesOperationEquals(rank3, matrix, expected);
        assertTimesInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{1, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 4, 9, 16}, new long[]{1, 2, 2});

        assertTimesOperationEquals(rank3, matrix, expected);
        assertTimesInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor rank4 = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8,
            4, 3, 2, 1, 7, 5, 8, 6
        }, new long[]{2, 2, 2, 2});

        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2}).reshape(2, 2, 1, 1);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 10, 12, 14, 16,
            12, 9, 6, 3, 28, 20, 32, 24
        }, new long[]{2, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyRank1AndMatrix() {

        /*
          a = np.array([1, 2, 3]).reshape(3)
          b = np.array([1, 2, 3, 5, 6, 7]).reshape(2, 3)
          ab = a * b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank4 = DoubleTensor.create(new double[]{
            1, 2, 3
        }, new long[]{3});

        DoubleTensor matrix = DoubleTensor.create(new double[]{
            1, 2, 3,
            5, 6, 7
        }, new long[]{2, 3});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 4, 9,
            5, 12, 21
        }, new long[]{2, 3});


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

        DoubleTensor rank5 = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
        }, new long[]{2, 2, 2, 2, 2});

        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2}).reshape(2, 2, 1, 1, 1);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 8, 6, 4, 2, 14, 10, 16, 12,
            18, 9, 6, 27, 9, 12, 21, 18, 24, 8, 20, 16, 0, 8, 4, 12
        }, new long[]{2, 2, 2, 2, 2});

        assertTimesOperationEquals(rank5, matrix, expected);
        assertTimesInPlaceOperationEquals(rank5, matrix, expected);
    }

    @Test
    public void canBroadcastScalarToMatrix() {

        DoubleTensor left = DoubleTensor.scalar(5);
        DoubleTensor right = DoubleTensor.arange(0, 10).reshape(2, 5);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            0, 5, 10, 15, 20,
            25, 30, 35, 40, 45
        }, new long[]{2, 5});

        assertTimesOperationEquals(left, right, expected);
        assertTimesInPlaceOperationEquals(left, right, expected);
    }

    @Test
    public void canBroadcastMatrixToScalar() {

        DoubleTensor left = DoubleTensor.arange(0, 10).reshape(2, 5);
        DoubleTensor right = DoubleTensor.scalar(5);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            0, 5, 10, 15, 20,
            25, 30, 35, 40, 45
        }, new long[]{2, 5});

        assertTimesOperationEquals(left, right, expected);
        assertTimesInPlaceOperationEquals(left, right, expected);
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

        DoubleTensor x = DoubleTensor.zeros(new long[]{2, 2, 2, 2});
        DoubleTensor y = DoubleTensor.create(new double[]{1, 0, 1, 0}, new long[]{2, 2});

        DoubleTensor diff = x.plus(y);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new long[]{2, 2, 2, 2});

        assertEquals(expected, diff);
    }

    @Test
    public void canSuperBroadcastInPlace() {
        DoubleTensor x = DoubleTensor.zeros(new long[]{2, 2, 2, 2});
        DoubleTensor y = DoubleTensor.create(new double[]{1, 0, 1, 0}, new long[]{2, 2});

        DoubleTensor diff = x.plusInPlace(y);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        }, new long[]{2, 2, 2, 2});

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

        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{3, 1});
        DoubleTensor s = DoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        DoubleTensor diff = s.plus(x);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastAddVector() {

        /*
          x = np.array([1,2,3]).reshape(3)
          s = np.array([-5, -5, -5,
            -2, -2, -2,
            -3, -3, -3,
            -7, -7, -7,
            -8, -8, -8]).reshape(3,5)
          sx = s + x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{3});
        DoubleTensor s = DoubleTensor.create(new double[]{
            -5, -5, -5,
            -2, -2, -2,
            -3, -3, -3,
            -7, -7, -7,
            -8, -8, -8
        }, new long[]{5, 3});

        DoubleTensor result = s.plus(x);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            -4, -3, -2,
            -1, 0, 1,
            -2, -1, 0,
            -6, -5, -4,
            -7, -6, -5
        }, new long[]{5, 3});

        assertEquals(expected, result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void canBroadcastAddToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a + b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2, 1});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{2, 3, 5, 6, 4, 5, 7, 8}, new long[]{2, 2, 2});

        assertPlusOperationEquals(rank3, matrix, expected);
        assertPlusInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{1, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{2, 4, 6, 8}, new long[]{1, 2, 2});

        assertPlusOperationEquals(rank3, matrix, expected);
        assertPlusInPlaceOperationEquals(rank3, matrix, expected);
    }

    @Test
    public void canBroadcastSubtractVector() {

        /*
          x = np.array([-1,-2,-3]).reshape(3)
          s = np.array([-5, -2, -3, -7, -8, -5, -2, -3, -7, -8, -5, -2, -3, -7, -8]).reshape(3,5)
          sx = s - x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = DoubleTensor.create(new double[]{-1, -2, -3}, new long[]{3});
        DoubleTensor s = DoubleTensor.create(new double[]{
            -5, -2, -3,
            -7, -8, -5,
            -2, -3, -7,
            -8, -5, -2,
            -3, -7, -8
        }, new long[]{5, 3});

        DoubleTensor diff = s.minus(x);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            -4, 0, 0,
            -6, -6, -2,
            -1, -1, -4,
            -7, -3, 1,
            -2, -5, -5
        }, new long[]{5, 3});

        assertEquals(expected, diff);
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

        DoubleTensor x = DoubleTensor.create(new double[]{-1, -2, -3}, new long[]{3, 1});
        DoubleTensor s = DoubleTensor.create(new double[]{
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8,
            -5, -2, -3, -7, -8
        }, new long[]{3, 5});

        DoubleTensor diff = s.minus(x);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            -4, -1, -2, -6, -7,
            -3, 0, -1, -5, -6,
            -2, 1, 0, -4, -5
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test(expected = IllegalArgumentException.class)
    public void canBroadcastSubtractToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([1, 2, 3, 4]).reshape(2,2,1)
          b = np.array([1, 2, 3, 4]).reshape(2,2)
          ab = a - b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2, 1});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{0, -1, -1, -2, 2, 1, 1, 0}, new long[]{2, 2, 2});

        assertMinusOperationEquals(rank3, matrix, expected);
        assertMinusInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{1, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 0, 0, 0}, new long[]{1, 2, 2});

        assertMinusOperationEquals(rank3, matrix, expected);
        assertMinusInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{3, 1});
        DoubleTensor s = DoubleTensor.create(new double[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{3, 5});

        DoubleTensor diff = s.div(x);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            5 / 1.0, 2 / 1.0, 3 / 1.0, 7 / 1.0, 8 / 1.0,
            5 / 2.0, 2 / 2.0, 3 / 2.0, 7 / 2.0, 8 / 2.0,
            5 / 3.0, 2 / 3.0, 3 / 3.0, 7 / 3.0, 8 / 3.0
        }, new long[]{3, 5});

        assertEquals(expected, diff);
    }

    @Test
    public void canBroadcastDivideVector() {

        /*
          x = np.array([1,2,3]).reshape(3)
          s = np.array([5, 5, 5,
            2, 2, 2,
            3, 3, 3,
            7, 7, 7,
            8, 8, 8]).reshape(5,3)
          sx = s / x
          print(sx)
          print(np.shape(sx))
         */

        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{3});
        DoubleTensor s = DoubleTensor.create(new double[]{
            5, 5, 5,
            2, 2, 2,
            3, 3, 3,
            7, 7, 7,
            8, 8, 8
        }, new long[]{5, 3});

        DoubleTensor diff = s.div(x);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            5 / 1.0, 5 / 2.0, 5 / 3.0,
            2 / 1.0, 2 / 2.0, 2 / 3.0,
            3 / 1.0, 3 / 2.0, 3 / 3.0,
            7 / 1.0, 7 / 2.0, 7 / 3.0,
            8 / 1.0, 8 / 2.0, 8 / 3.0
        }, new long[]{5, 3});

        assertEquals(expected, diff);
    }

    @Test(expected = IllegalArgumentException.class)
    public void canBroadcastDivideToHigherRankWithDimensionsOfOneTailing() {

        /*
          a = np.array([10, 20, 30, 40]).reshape(2,2,1)
          b = np.array([1, 2, 5, 10]).reshape(2,2)
          ab = a / b
          print(ab)
          print(np.shape(ab))
         */

        DoubleTensor rank3 = DoubleTensor.create(new double[]{10, 20, 30, 40}, new long[]{2, 2, 1});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{10., 5., 4., 2., 30., 15., 8., 4.}, new long[]{2, 2, 2});

        assertDivideOperationEquals(rank3, matrix, expected);
        assertDivideInPlaceOperationEquals(rank3, matrix, expected);
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

        DoubleTensor rank3 = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{1, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor expected = DoubleTensor.create(new double[]{1., 1., 1., 1.}, new long[]{1, 2, 2});

        assertDivideOperationEquals(rank3, matrix, expected);
        assertDivideInPlaceOperationEquals(rank3, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new long[]{4, 2, 2, 2});


        assertTimesOperationEquals(rank4, matrix, expected);
        assertTimesInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMultiplyDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(4, 2, 2, 2);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new long[]{4, 2, 2, 2});


        assertTimesOperationEquals(matrix, rank4, expected);
        assertTimesInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.zeros(new long[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new long[]{4, 2, 2, 2});

        assertPlusOperationEquals(rank4, matrix, expected);
        assertPlusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastPlusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.zeros(new long[]{4, 2, 2, 2});
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4,
        }, new long[]{4, 2, 2, 2});

        assertPlusOperationEquals(matrix, rank4, expected);
        assertPlusInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new long[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
            10, 5, 2, 1, 10, 5, 2, 1,
        }, new long[]{4, 2, 2, 2});

        assertDivideOperationEquals(rank4, matrix, expected);
        assertDivideInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastDivideDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new long[]{4, 2, 2, 2}).times(10.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 5, 10}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
            .1, .2, .5, 1, .1, .2, .5, 1,
        }, new long[]{4, 2, 2, 2});

        assertDivideOperationEquals(matrix, rank4, expected);
        assertDivideInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsBigToSmall() {
        DoubleTensor rank4 = DoubleTensor.ones(new long[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1,
            4, 3, 2, 1, 4, 3, 2, 1
        }, new long[]{4, 2, 2, 2});

        assertMinusOperationEquals(rank4, matrix, expected);
        assertMinusInPlaceOperationEquals(rank4, matrix, expected);
    }

    @Test
    public void canBroadcastMinusDifferentRankedTensorsSmallToBig() {
        DoubleTensor rank4 = DoubleTensor.ones(new long[]{4, 2, 2, 2}).times(5.);
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1,
            -4, -3, -2, -1, -4, -3, -2, -1
        }, new long[]{4, 2, 2, 2});

        assertMinusOperationEquals(matrix, rank4, expected);
        assertMinusInPlaceOperationEquals(matrix, rank4, expected);
    }

    @Test
    public void canBroadcastPow() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        DoubleTensor exponent = DoubleTensor.create(2, 3);

        DoubleTensor expected = DoubleTensor.create(new double[]{1, 8, 9, 64}, 2, 2);

        assertPowOperationEquals(matrix, exponent, expected);
        assertPowInPlaceOperationEquals(matrix, exponent, expected);
    }

}
