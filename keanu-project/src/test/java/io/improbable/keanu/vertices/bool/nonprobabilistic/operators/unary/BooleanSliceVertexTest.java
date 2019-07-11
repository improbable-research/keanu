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
        BooleanVertex rowOne = matrixA.slice(0, 0);

        Assert.assertArrayEquals(new double[]{1, 1, 0}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        BooleanVertex rowTwo = matrixA.slice(0, 1);

        Assert.assertArrayEquals(new double[]{0, 1, 1}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        BooleanVertex columnOne = matrixA.slice(1, 0);

        Assert.assertArrayEquals(new double[]{1, 0}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        BooleanVertex columnTwo = matrixA.slice(1, 1);

        Assert.assertArrayEquals(new double[]{1, 1}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        BooleanVertex columnThree = matrixA.slice(1, 2);

        Assert.assertArrayEquals(new double[]{0, 1}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3() {
        BooleanVertex cube = new ConstantBooleanVertex(false);
        cube.setValue(BooleanTensor.create(new boolean[]{true, true, false, false, true, true, false, false}, 2, 2, 2));

        BooleanVertex dimenZeroFace = cube.slice(0, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        BooleanVertex dimenOneFace = cube.slice(1, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        BooleanVertex dimenTwoFace = cube.slice(2, 0);
        Assert.assertArrayEquals(new double[]{1, 0, 1, 0}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

}
