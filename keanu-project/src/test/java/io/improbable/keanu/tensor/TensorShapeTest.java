package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TensorShapeTest {

    @Test
    public void canCalculateRowFirstStride() {
        long[] shape = new long[]{2, 3, 7, 4};

        assertArrayEquals(new long[]{84, 28, 4, 1}, TensorShape.getRowFirstStride(shape));
    }

    @Test
    public void canCalculateLength() {
        long[] shape = new long[]{2, 3, 7, 4};
        assertEquals(168, TensorShape.getLength(shape));
    }

    @Test
    public void canGetDimensionRange() {
        int[] actual = TensorShape.dimensionRange(2, 5);
        int[] expected = new int[]{2, 3, 4};
        assertArrayEquals(actual, expected);
    }

    @Test
    public void canGetAbsoluteDimensionsFromRelative() {
        int[] actual = TensorShape.setToAbsoluteDimensions(3, new int[]{-2, 1, 2});
        int[] expected = new int[]{1, 1, 2};
        assertArrayEquals(actual, expected);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidNegativeAbsoluteDimensionsFromRelative() {
        TensorShape.setToAbsoluteDimensions(3, new int[]{-4, 1, 2});
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidPositiveAbsoluteDimensionsFromRelative() {
        TensorShape.setToAbsoluteDimensions(3, new int[]{3, 1, 2});
    }

    @Test
    public void canGetShapeIndices() {
        Assert.assertArrayEquals(new long[]{2, 2}, TensorShape.getShapeIndices(new long[]{5, 4}, new long[]{4, 1}, 10));
        Assert.assertArrayEquals(new long[]{4, 3}, TensorShape.getShapeIndices(new long[]{5, 4}, new long[]{4, 1}, 19));
        Assert.assertArrayEquals(new long[]{1, 1, 1}, TensorShape.getShapeIndices(new long[]{3, 3, 3}, new long[]{9, 3, 1}, 13));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfFlatIndexOverflowsShape() {
        long[] shapeIndices = TensorShape.getShapeIndices(new long[]{5, 4}, new long[]{4, 1}, 20);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfFlatIndexUnderflowsShape() {
        long[] shapeIndices = TensorShape.getShapeIndices(new long[]{5, 4}, new long[]{4, 1}, -1);
    }

    @Test
    public void canGetElementFromShapeIndices() {
        long[] shape = new long[]{2, 2, 3};

        double[] buffer = new double[(int) TensorShape.getLength(shape)];
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = i;
        }

        DoubleTensor tensor = DoubleTensor.create(buffer, shape);
        long[] stride = TensorShape.getRowFirstStride(shape);

        for (int i = 0; i < buffer.length; i++) {
            long[] indexOfi = TensorShape.getShapeIndices(shape, stride, i);
            assertEquals(i, tensor.getValue(indexOfi), 1e-10);
        }
    }

    @Test
    public void canIncrementIndexForSingleDimensionOfMatrix() {
        long[] shape = new long[]{2, 3};

        long[][] expected1 = new long[][]{
            new long[]{0, 1},
            new long[]{0, 2}
        };

        assertIndexIncrementResults(shape, new long[]{0, 0}, new int[]{1}, expected1);

        long[][] expected2 = new long[][]{
            new long[]{1, 0}
        };

        assertIndexIncrementResults(shape, new long[]{0, 0}, new int[]{0}, expected2);
    }

    @Test
    public void canIncrementIndexForRank3() {
        long[] shape = new long[]{2, 3, 3};
        long[] index = new long[]{0, 0, 2};
        int[] dimensionOrder = new int[]{1, 0};

        long[][] expected = new long[][]{
            new long[]{0, 1, 2},
            new long[]{0, 2, 2},
            new long[]{1, 0, 2},
            new long[]{1, 1, 2},
            new long[]{1, 2, 2}
        };

        assertIndexIncrementResults(shape, index, dimensionOrder, expected);
    }

    private void assertIndexIncrementResults(long[] shape, long[] index, int[] dimensionOrder, long[][] expected) {
        for (int i = 0; i < expected.length; i++) {
            boolean result = TensorShape.incrementIndexByShape(shape, index, dimensionOrder);
            assertArrayEquals(index, expected[i]);
            assertTrue(result);
        }

        assertFalse(TensorShape.incrementIndexByShape(shape, index, dimensionOrder));
    }


}
