package io.improbable.keanu.tensor;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class TensorShapeValidationTest {

    private long[] scalar1 = new long[0];
    private long[] scalar2 = new long[0];
    private long[] twoByTwo1 = new long[]{2, 2};
    private long[] twoByTwo2 = new long[]{2, 2};
    private long[] twoByThree = new long[]{2, 3};

    @Test
    public void suggestSingleNonScalarShape() {
        long[] shapeProposal = TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne(scalar1, twoByTwo1, scalar2);
        assertArrayEquals(new long[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestMatchingNonScalarShape() {
        long[] shapeProposal = TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne(scalar1, twoByTwo1, twoByTwo2);
        assertArrayEquals(new long[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestScalar() {
        long[] shapeProposal = TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne(scalar1, scalar2);
        assertArrayEquals(Tensor.SCALAR_SHAPE, shapeProposal);
    }

    @Test
    public void suggestHighestRankLengthOne() {
        long[] shapeProposal = TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne(new long[]{1, 1}, new long[]{1, 1, 1}, new long[]{1});
        assertArrayEquals(new long[]{1, 1, 1}, shapeProposal);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMultipleNonScalars() {
        TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne(scalar1, twoByThree, twoByTwo1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsShapeNotMatchingParent() {
        TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne(new long[]{2, 4}, twoByTwo1, scalar1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMoreThanSingleNonScalarParent() {
        TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne(new long[]{2, 2}, twoByTwo1, twoByThree);
    }

    @Test
    public void acceptsMatchingParentsShape() {
        TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne(new long[]{2, 2}, twoByTwo1, scalar1);
    }

    @Test
    public void acceptsNonScalarToScalarParentShape() {
        TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne(new long[]{2, 4}, scalar2, scalar1);
    }

    @Test
    public void checkSquareMatrixAcceptsSquareMatrices() {
        TensorShapeValidation.checkShapeIsSquareMatrix(new long[]{3, 3});
    }

    @Test(expected = IllegalArgumentException.class)
    public void checkSquareMatrixFailsOnNonMatrices() {
        TensorShapeValidation.checkShapeIsSquareMatrix(new long[]{3, 3, 3});
    }

    @Test(expected = IllegalArgumentException.class)
    public void checkSquareMatrixFailsOnNonSquareMatrices() {
        TensorShapeValidation.checkShapeIsSquareMatrix(new long[]{3, 2});
    }
}
