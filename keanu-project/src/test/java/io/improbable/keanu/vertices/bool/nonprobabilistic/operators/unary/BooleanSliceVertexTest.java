package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;


public class BooleanSliceVertexTest {

    private BooleanVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, false, false, true, true}, 2, 3));
    }

    @Test
    public void canGetTensorAlongDimensionOfRank2() {
        BooleanSliceVertex rowOne = new BooleanSliceVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new double[]{1, 1, 0}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        BooleanSliceVertex rowTwo = new BooleanSliceVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new double[]{0, 1, 1}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        BooleanSliceVertex columnOne = new BooleanSliceVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new double[]{1, 0}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        BooleanSliceVertex columnTwo = new BooleanSliceVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new double[]{1, 1}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        BooleanSliceVertex columnThree = new BooleanSliceVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new double[]{0, 1}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3() {
        BooleanVertex cube = new ConstantBooleanVertex(false);
        cube.setValue(BooleanTensor.create(new boolean[]{true, true, false, false, true, true, false, false}, 2, 2, 2));

        BooleanSliceVertex dimenZeroFace = new BooleanSliceVertex(cube, 0, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        BooleanSliceVertex dimenOneFace = new BooleanSliceVertex(cube, 1, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        BooleanSliceVertex dimenTwoFace = new BooleanSliceVertex(cube, 2, 0);
        Assert.assertArrayEquals(new double[]{1, 0, 1, 0}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

}
