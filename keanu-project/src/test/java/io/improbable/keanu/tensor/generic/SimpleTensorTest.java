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

    Tensor<Something> tensorA;
    DoubleTensor maskA;

    @Before
    public void setup() {
        tensorA = new GenericTensor<>(
            new Something[]{
                Something.A, Something.B, Something.B,
                Something.C, Something.D, Something.B,
                Something.D, Something.A, Something.C
            },
            new int[]{3, 3}
        );
        maskA = DoubleTensor.create(
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
        assertEquals(Something.D, tensorA.getValue(1, 1));
    }

    @Test
    public void canSetRandomAccessValue() {
        assertEquals(Something.D, tensorA.getValue(1, 1));

        tensorA.setValue(Something.A, 1, 1);

        assertEquals(Something.A, tensorA.getValue(1, 1));
    }

    @Test
    public void canReshape() {
        Tensor<Something> reshapedSomething = tensorA.reshape(9, 1);

        assertArrayEquals(new int[]{9, 1}, reshapedSomething.getShape());
        assertArrayEquals(tensorA.asFlatArray(), reshapedSomething.asFlatArray());
    }

    @Test
    public void canTake() {
        ConstantGenericVertex<Something> somethingVertex = new ConstantGenericVertex(tensorA);

        GenericTakeVertex<Something> take = new GenericTakeVertex(somethingVertex, 0, 0);

        assertEquals(Something.A, take.getValue().scalar());
    }

    @Test
    public void canSliceRankTwoTensor() {
        Tensor<Something> taddedSomethingRow = tensorA.slice(0, 1);
        assertArrayEquals(new int[]{1, 3}, taddedSomethingRow.getShape());
        assertArrayEquals(new Something[]{Something.C, Something.D, Something.B}, taddedSomethingRow.asFlatArray());

        Tensor<Something> taddedSomethingColumn = tensorA.slice(1, 1);
        assertArrayEquals(new int[]{3, 1}, taddedSomethingColumn.getShape());
        assertArrayEquals(new Something[]{Something.B, Something.D, Something.A}, taddedSomethingColumn.asFlatArray());
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        DoubleTensor mask = DoubleTensor.scalar(1.);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensorA.getLength() + ", mask length: " + mask.getLength());

        tensorA.setWithMaskInPlace(mask, Something.A);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        Tensor<Something> tensor = Tensor.scalar(Something.A);
        DoubleTensor mask = DoubleTensor.create(new double[] {1., 1., 1., 1.}, 2, 2);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, Something.B);
    }

    @Test
    public void canSetWithMaskIfLengthOfMaskAndTensorAreEqual() {
        Tensor<Something> result = tensorA.setWithMask(maskA, Something.B);

        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            Something.C, Something.D, Something.B,
            Something.B, Something.B, Something.B
        }, result.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfLengthOfMaskAndNonScalarTensorAreEqual() {
        tensorA.setWithMaskInPlace(maskA, Something.B);
        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            Something.C, Something.D, Something.B,
            Something.B, Something.B, Something.B
        }, tensorA.asFlatArray());
    }

    @Test
    public void canSetWithMaskInPlaceIfMaskAndTensorAreScalars() {
        Tensor<Something> scalarTensor = new GenericTensor<>(Something.A);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        scalarTensor.setWithMaskInPlace(mask, Something.B);

        assertTrue(scalarTensor.isScalar());
        assertEquals(Something.B, scalarTensor.scalar());
    }

    @Test
    public void canSetWithMaskInPlaceIfTensorIsShapePlaceHolder() {
        Tensor<Something> shapePlaceHolder = new GenericTensor<>(new int[] {3, 3});

        assertTrue(shapePlaceHolder.isShapePlaceholder());
        shapePlaceHolder.setWithMaskInPlace(maskA, Something.B);
        assertFalse(shapePlaceHolder.isShapePlaceholder());

        assertArrayEquals(new Something[] {
            Something.B, Something.B, Something.B,
            null, null, null,
            Something.B, Something.B, Something.B
        }, shapePlaceHolder.asFlatArray());
    }

}
