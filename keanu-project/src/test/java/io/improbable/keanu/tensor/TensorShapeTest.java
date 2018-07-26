package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
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

    @Test
    public void canGetElementFromShapeIndices() {
        int[] shape = new int[]{2, 2, 3};

        double[] buffer = new double[(int) TensorShape.getLength(shape)];
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = i;
        }

        DoubleTensor tensor = DoubleTensor.create(buffer, shape);
        int[] stride = TensorShape.getRowFirstStride(shape);

        for (int i = 0; i < buffer.length; i++) {
            int[] indexOfi = TensorShape.getShapeIndices(shape, stride, i);
            assertEquals(i, tensor.getValue(indexOfi), 1e-10);
        }
    }
}
