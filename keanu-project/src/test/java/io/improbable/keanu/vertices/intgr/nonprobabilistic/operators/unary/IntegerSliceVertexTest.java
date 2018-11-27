package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IntegerSliceVertexTest {

    private IntegerVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantIntegerVertex(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }

    @Test
    public void canGetTensorAlongDimensionOfRank2() {
        IntegerSliceVertex rowOne = new IntegerSliceVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new int[]{1, 2, 3}, rowOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{1, 3}, rowOne.getShape());

        IntegerSliceVertex rowTwo = new IntegerSliceVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new int[]{4, 5, 6}, rowTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{1, 3}, rowTwo.getShape());

        IntegerSliceVertex columnOne = new IntegerSliceVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new int[]{1, 4}, columnOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 1}, columnOne.getShape());

        IntegerSliceVertex columnTwo = new IntegerSliceVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new int[]{2, 5}, columnTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 1}, columnTwo.getShape());

        IntegerSliceVertex columnThree = new IntegerSliceVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new int[]{3, 6}, columnThree.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 1}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3() {
        IntegerVertex cube = new ConstantIntegerVertex(0);
        cube.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        IntegerSliceVertex dimenZeroFace = new IntegerSliceVertex(cube, 0, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        IntegerSliceVertex dimenOneFace = new IntegerSliceVertex(cube, 1, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        IntegerSliceVertex dimenTwoFace = new IntegerSliceVertex(cube, 2, 0);
        Assert.assertArrayEquals(new int[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

}
