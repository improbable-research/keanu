package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import org.junit.rules.ExpectedException;

public class SimpleTensorTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

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
    public void canTake() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        ConstantGenericVertex<Something> somethingVertex = new ConstantGenericVertex(somethingTensor);

        GenericTakeVertex<Something> take = new GenericTakeVertex(somethingVertex, 0, 0);

        assertEquals(Something.A, take.getValue().scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {

        Tensor<Something> somethingTensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );

        Tensor<Something> taddedSomethingRow = somethingTensor.slice(0, 1);
        assertArrayEquals(new int[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = somethingTensor.slice(1, 1);
        assertArrayEquals(new int[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

    @Test
    public void isNullReturnsTrueIfNoDataIsSet() {
        GenericTensor<Something> tensor = new GenericTensor<>(new int[] {1, 1});
        assertTrue(tensor.isNull());
    }

    @Test
    public void isNullReturnsFalseIfDataIsSet() {
        GenericTensor<Something> tensor = new GenericTensor<>(new Something[] {Something.A}, new int[] {1, 1});
        assertFalse(tensor.isNull());
    }

    @Test
    public void canSetWhereEqualMatrix() {
         GenericTensor<Something> tensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.A,
                Something.C, Something.D,
            },
            new int[]{2, 2});
         DoubleTensor mask = tensor.equalsMask(Something.A);
         GenericTensor result = tensor.setWithMaskInPlace(mask, Something.B);

         assertArrayEquals(new Something[]{Something.B, Something.B, Something.C, Something.D}, result.asFlatArray());
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        GenericTensor<Something> tensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3});
        DoubleTensor mask = DoubleTensor.scalar(1);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, Something.C);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        GenericTensor<Something> tensor = new GenericTensor<>(Something.A);
        DoubleTensor mask = DoubleTensor.create(new double[] {1., 1., 1., 1.}, 2, 2);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, Something.C);
    }

}
