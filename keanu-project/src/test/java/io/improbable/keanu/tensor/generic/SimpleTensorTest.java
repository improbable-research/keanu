package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SimpleTensorTest {

    enum Something {
        A, B, C, D
    }

    @Test
    public void canGetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));

        somethingTensor.setValue(Something.A, 1, 1);

        assertEquals(Something.A, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canReshape() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        Tensor<Something> reshapedSomething = somethingTensor.reshape(9, 1);

        assertArrayEquals(new int[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(somethingTensor.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canTAD() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        Tensor<Something> taddedSomethingRow = somethingTensor.alongDimension(0, 1);
        assertArrayEquals(new int[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = somethingTensor.alongDimension(1, 1);
        assertArrayEquals(new int[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

}
