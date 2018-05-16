package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class ProbabilisticVertexShapingTest {

    private DoubleTensor scalar0 = Nd4jDoubleTensor.scalar(0.0);
    private DoubleTensor scalar3 = Nd4jDoubleTensor.scalar(3.0);
    private DoubleTensor twoByTwo1 = new Nd4jDoubleTensor(new double[]{1, 1, 1, 1}, new int[]{2, 2});
    private DoubleTensor twoByTwo2 = new Nd4jDoubleTensor(new double[]{2, 2, 2, 2}, new int[]{2, 2});
    private DoubleTensor twoByThree = new Nd4jDoubleTensor(new double[]{2, 3, 4, 5, 6, 7}, new int[]{2, 3});

    @Test
    public void suggestSingleNonScalarShape() {
        int[] shapeProposal = ProbabilisticVertexShaping.getShapeProposal(scalar0, twoByTwo1, scalar3);
        assertArrayEquals(new int[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestMatchingNonScalarShape() {
        int[] shapeProposal = ProbabilisticVertexShaping.getShapeProposal(scalar0, twoByTwo1, twoByTwo2);
        assertArrayEquals(new int[]{2, 2}, shapeProposal);
    }

    @Test
    public void suggestScalar() {
        int[] shapeProposal = ProbabilisticVertexShaping.getShapeProposal(scalar0, scalar3);
        assertArrayEquals(new int[]{1, 1}, shapeProposal);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMultipleNonScalars() {
        ProbabilisticVertexShaping.getShapeProposal(scalar0, twoByThree, twoByTwo1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsShapeNotMatchingParent() {
        ProbabilisticVertexShaping.checkParentShapes(new int[]{2, 4}, twoByTwo1, scalar0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMoreThanSingleNonScalarParent() {
        ProbabilisticVertexShaping.checkParentShapes(new int[]{2, 2}, twoByTwo1, twoByThree);
    }

    @Test
    public void acceptsMatchingParentsShape() {
        ProbabilisticVertexShaping.checkParentShapes(new int[]{2, 2}, twoByTwo1, scalar0);
    }

    @Test
    public void acceptsNonScalarToScalarParentShape() {
        ProbabilisticVertexShaping.checkParentShapes(new int[]{2, 4}, scalar3, scalar0);
    }
}
