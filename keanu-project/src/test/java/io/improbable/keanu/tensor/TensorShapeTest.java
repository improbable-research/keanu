package io.improbable.keanu.tensor;

import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TensorShapeTest {

    @Test
    public void canCalculateRowFirstStride() {
        int[] shape = new int[]{2, 3, 7, 4};

        assertArrayEquals(new int[]{84, 28, 4, 1}, TensorShape.getRowFirstStride(shape));
    }

    @Test
    public void canCalculateLength() {
        int[] shape = new int[]{2, 3, 7, 4};
        assertEquals(168, TensorShape.getLength(shape));
    }

    @Test
    public void canGetDimensionRange() {
        int[] actual = TensorShape.dimensionRange(2, 5);
        int[] expected = new int[]{2, 3, 4};
        assertArrayEquals(actual, expected);
    }

    @Test
    public void canGetShapeIndices() {
        Assert.assertArrayEquals(new int[]{2, 2}, TensorShape.getShapeIndices(new int[]{5, 4}, new int[]{4, 1}, 10));
        Assert.assertArrayEquals(new int[]{1, 1, 1}, TensorShape.getShapeIndices(new int[]{3, 3, 3}, new int[]{9, 3, 1}, 13));
    }
}
