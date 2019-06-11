package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class GenericTensorTest {

    enum Something {
        A, B, C, D
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        String value = "foo";
        String otherValue = "bar";
        GenericTensor<String> allTheSame = new GenericTensor<>(new long[]{2, 3}, value);
        GenericTensor<String> notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canGetRandomAccessValue() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {

        GenericTensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        assertEquals(Something.D, somethingTensor.getValue(1, 1));

        somethingTensor.setValue(Something.A, 1, 1);

        assertEquals(Something.A, somethingTensor.getValue(1, 1));
    }

    @Test
    public void canReshape() {

        GenericTensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        GenericTensor<Something> reshapedSomething = somethingTensor.reshape(9, 1);

        assertArrayEquals(new long[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(somethingTensor.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canTake() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        assertEquals(Something.A, somethingTensor.take(0, 0).scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        Tensor<Something> taddedSomethingRow = somethingTensor.slice(0, 1);
        assertArrayEquals(new long[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = somethingTensor.slice(1, 1);
        assertArrayEquals(new long[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

    @Test
    public void canPermute() {
        GenericTensor<Something> somethingTensor = GenericTensor.create(
            Something.A, Something.B, Something.B,
            Something.C, Something.D, Something.B,
            Something.D, Something.A, Something.C
        ).reshape(3, 3);

        GenericTensor<Something> permute = somethingTensor.permute(1, 0);

        GenericTensor<Something> expected = GenericTensor.create(
            Something.A, Something.C, Something.D,
            Something.B, Something.D, Something.A,
            Something.B, Something.B, Something.C
        ).reshape(3, 3);

        assertThat(permute, valuesAndShapesMatch(expected));
    }

    @Test
    public void canGetFlatIntegers() {
        GenericTensor<Integer> somethingTensor = GenericTensor.create(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ).reshape(3, 3);

        int[] actual = somethingTensor.asFlatIntegerArray();

        assertArrayEquals(actual, new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    @Test
    public void canGetFlatDoubles() {
        GenericTensor<Double> somethingTensor = GenericTensor.create(
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ).reshape(3, 3);

        double[] actual = somethingTensor.asFlatDoubleArray();

        assertArrayEquals(actual, new double[]{1., 2., 3., 4., 5., 6., 7., 8., 9.}, 1e-8);
    }

}