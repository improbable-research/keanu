package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.TensorShapeValidation;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class ProbabilisticVertexShapingTest {

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
        assertArrayEquals(new long[]{}, shapeProposal);
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
}
