package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Test;

import java.util.Arrays;

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
        GenericTensor<String> allTheSame = GenericTensor.createFilled(value, new long[]{2, 3});
        GenericTensor<String> notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue().scalar(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canGetRandomAccessValue() {

        GenericTensor<Something> somethingTensor = GenericTensor.create(
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

        GenericTensor<Something> somethingTensor = GenericTensor.create(
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

        GenericTensor<Something> somethingTensor = GenericTensor.create(
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

        GenericTensor<Something> somethingTensor = GenericTensor.create(
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

        GenericTensor<Something> somethingTensor = GenericTensor.create(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new long[]{3, 3}
        );

        GenericTensor<Something> taddedSomethingRow = somethingTensor.slice(0, 1);
        assertArrayEquals(new long[]{3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        GenericTensor<Something> taddedSomethingColumn = somethingTensor.slice(1, 1);
        assertArrayEquals(new long[]{3}, taddedSomethingColumn.getShape());
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
    public void canGetFlatArray() {
        GenericTensor<Integer> somethingTensor = GenericTensor.create(
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ).reshape(3, 3);

        Integer[] actual = somethingTensor.asFlatArray();

        assertArrayEquals(actual, new Integer[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    }

    @Test
    public void canConvertGenericTypeByApply() {
        GenericTensor<Double> a = GenericTensor.create(1., 2., 3.);
        GenericTensor<Double> b = GenericTensor.create(4., 5., 6.);

        GenericTensor<String> actual = a.apply(b, (l, r) -> (l + r) + "");
        GenericTensor<String> expected = GenericTensor.create("5.0", "7.0", "9.0");

        assertThat(actual, valuesAndShapesMatch(expected));
    }

    @Test
    public void canApplyMultiplyWithBroadcast() {
        GenericTensor<Double> a = GenericTensor.create(
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ).reshape(3, 3);

        GenericTensor<Double> actual = a.apply(DoubleTensor.create(0.5, 0.25, 0.1), (l, r) -> l * r);

        GenericTensor<Double> expected = GenericTensor.create(
            0.5, 0.5, 0.3,
            2., 1.25, 0.6,
            3.5, 2., 0.9
        ).reshape(3, 3);

        assertArrayEquals(
            Arrays.stream(actual.asFlatArray()).mapToDouble(x -> x).toArray(),
            Arrays.stream(expected.asFlatArray()).mapToDouble(x -> x).toArray(),
            1e-8
        );
    }

    @Test
    public void canApplyStringConcatWithBroadcast() {
        GenericTensor<String> a = GenericTensor.create(
            "a", "b", "c",
            "d", "e", "f",
            "g", "h", "i"
        ).reshape(3, 3);

        GenericTensor<String> actualFromLeft = a.apply(GenericTensor.create("x", "y", "z"), (l, r) -> l + r);

        GenericTensor<String> expectedFromLeft = GenericTensor.create(
            "ax", "by", "cz",
            "dx", "ey", "fz",
            "gx", "hy", "iz"
        ).reshape(3, 3);

        assertArrayEquals(actualFromLeft.asFlatArray(), expectedFromLeft.asFlatArray());

        GenericTensor<String> actualFromRight = GenericTensor.create("x", "y", "z").apply(a, (l, r) -> l + r);

        GenericTensor<String> expectedFromRight = GenericTensor.create(
            "xa", "yb", "zc",
            "xd", "ye", "zf",
            "xg", "yh", "zi"
        ).reshape(3, 3);

        assertArrayEquals(actualFromRight.asFlatArray(), expectedFromRight.asFlatArray());

        GenericTensor<String> actualByColumn = GenericTensor.create("x", "y", "z").reshape(3, 1).apply(a, (l, r) -> l + r);

        GenericTensor<String> expectedByColumn = GenericTensor.create(
            "xa", "xb", "xc",
            "yd", "ye", "yf",
            "zg", "zh", "zi"
        ).reshape(3, 3);

        assertArrayEquals(actualByColumn.asFlatArray(), expectedByColumn.asFlatArray());
    }

    @Test
    public void canBroadcastToShape() {
        GenericTensor<String> a = GenericTensor.create(
            "a", "b", "c"
        ).reshape(3);

        GenericTensor<String> expectedByRow = GenericTensor.create(
            "a", "b", "c",
            "a", "b", "c",
            "a", "b", "c"
        ).reshape(3, 3);

        assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expectedByRow));

        GenericTensor<String> expectedByColumn = GenericTensor.create(
            "a", "a", "a",
            "b", "b", "b",
            "c", "c", "c"
        ).reshape(3, 3);

        assertThat(a.reshape(3, 1).broadcast(3, 3), valuesAndShapesMatch(expectedByColumn));
    }

    @Test
    public void canDiagFromVector() {
        GenericTensor<String> expected = GenericTensor.create(new String[]{"1", null, null, null, "5", null, null, null, "9"}).reshape(3, 3);
        GenericTensor<String> actual = GenericTensor.create("1", "5", "9").diag();

        assertEquals(expected, actual);
    }

    @Test
    public void canDiagFromMatrix() {
        GenericTensor<String> actual = GenericTensor.create(new String[]{"1", "2", "3", "4", "5", "6", "7", "8", "9"}).reshape(3, 3).diagPart();
        GenericTensor<String> expected = GenericTensor.create("1", "5", "9");

        assertEquals(expected, actual);
    }

}