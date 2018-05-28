package io.improbable.keanu.tensor;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class TensorShapeValidationTest {

    private int[] scalar1 = new int[]{1, 1};
    private int[] scalar2 = new int[]{1, 1};
    private int[] twoByTwo1 = new int[]{2, 2};
    private int[] twoByTwo2 = new int[]{2, 2};
    private int[] twoByThree = new int[]{2, 3};

    @Test
    public void suggestSingleNonScalarShape() {
        int[] shapeProposal = TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(scalar1, twoByTwo1, scalar2);
        assertArrayEquals(new int[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestMatchingNonScalarShape() {
        int[] shapeProposal = TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(scalar1, twoByTwo1, twoByTwo2);
        assertArrayEquals(new int[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestScalar() {
        int[] shapeProposal = TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(scalar1, scalar2);
        assertArrayEquals(new int[]{1, 1}, shapeProposal);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMultipleNonScalars() {
        TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar(scalar1, twoByThree, twoByTwo1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsShapeNotMatchingParent() {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(new int[]{2, 4}, twoByTwo1, scalar1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMoreThanSingleNonScalarParent() {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(new int[]{2, 2}, twoByTwo1, twoByThree);
    }

    @Test
    public void acceptsMatchingParentsShape() {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(new int[]{2, 2}, twoByTwo1, scalar1);
    }

    @Test
    public void acceptsNonScalarToScalarParentShape() {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(new int[]{2, 4}, scalar2, scalar1);
    }
}
