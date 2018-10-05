package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import org.junit.Before;
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

    Tensor<Something> tensor;
    DoubleTensor mask;

    @Before
    public void setup() {
        tensor = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );
        mask = DoubleTensor.create(
            new double[] {
                1., 1., 1.,
                0., 0., 0.,
                1., 1., 1.,
            },
            3, 3
        );
    }

    @Test
    public void canGetRandomAccessValue() {
        assertEquals(Something.D, tensor.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {
        assertEquals(Something.D, tensor.getValue(1, 1));

        tensor.setValue(Something.A, 1, 1);

        assertEquals(Something.A, tensor.getValue(1, 1));
    }

    @Test
    public void canReshape() {
        Tensor<Something> reshapedSomething = tensor.reshape(9, 1);

        assertArrayEquals(new int[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(tensor.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canTake() {
        ConstantGenericVertex<Something> somethingVertex = new ConstantGenericVertex(tensor);

        GenericTakeVertex<Something> take = new GenericTakeVertex(somethingVertex, 0, 0);

        assertEquals(Something.A, take.getValue().scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {
        Tensor<Something> taddedSomethingRow = tensor.slice(0, 1);
        assertArrayEquals(new int[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = tensor.slice(1, 1);
        assertArrayEquals(new int[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        DoubleTensor smallMask = DoubleTensor.scalar(1.);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + smallMask.getLength());

        tensor.setWithMaskInPlace(smallMask, Something.A);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        Tensor<Something> smallTensor = Tensor.scalar(Something.A);
        DoubleTensor largerMask = DoubleTensor.create(new double[] {1., 1., 1., 1.}, 2, 2);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + smallTensor.getLength() + ", mask length: " + largerMask.getLength());

        smallTensor.setWithMaskInPlace(largerMask, Something.B);
    }

    @Test
    public void canSetWithMaskIfLengthOfMaskAndNonScalarTensorAreEqual() {
        Tensor<Something> result = tensor.setWithMask(mask, Something.B);

        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            Something.C, Something.D, Something.B,
            Something.B, Something.B, Something.B
        }, result.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfLengthOfMaskAndNonScalarTensorAreEqual() {
        tensor.setWithMaskInPlace(mask, Something.B);
        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            Something.C, Something.D, Something.B,
            Something.B, Something.B, Something.B
        }, tensor.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfMaskAndTensorAreScalars() {
        Tensor<Something> scalarTensor = new GenericTensor<>(Something.A);
        DoubleTensor scalarMask = DoubleTensor.scalar(1.);
        scalarTensor.setWithMaskInPlace(scalarMask, Something.B);

        assertTrue(scalarTensor.isScalar());
        assertEquals(Something.B, scalarTensor.scalar());
    }

    @Test
    public void canSetWithMaskInPlaceIfTensorIsNonScalarShapePlaceHolder() {
        Tensor<Something> shapePlaceHolder = new GenericTensor<>(new int[] {3, 3});

        assertFalse(shapePlaceHolder.isScalar());
        assertTrue(shapePlaceHolder.isShapePlaceholder());
        shapePlaceHolder.setWithMaskInPlace(mask, Something.B);
        assertFalse(shapePlaceHolder.isShapePlaceholder());

        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            null, null, null,
            Something.B, Something.B, Something.B
        }, shapePlaceHolder.asFlatArray());
    }


    @Test
    public void canSetWithMaskInPlaceIfTensorIsScalarShapePlaceHolder() {
        Tensor<Something> shapePlaceHolder = new GenericTensor<>(new int[] {1, 1});
        DoubleTensor scalarMask = DoubleTensor.scalar(1.);

        assertTrue(shapePlaceHolder.isScalar());
        assertTrue(shapePlaceHolder.isShapePlaceholder());
        shapePlaceHolder.setWithMaskInPlace(scalarMask, Something.B);
        assertFalse(shapePlaceHolder.isShapePlaceholder());

        assertArrayEquals(new Something[] {Something.B}, shapePlaceHolder.asFlatArray());
    }
}
