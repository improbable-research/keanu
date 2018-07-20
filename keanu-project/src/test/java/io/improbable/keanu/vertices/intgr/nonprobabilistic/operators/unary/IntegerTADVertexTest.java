package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IntegerTADVertexTest {

    private IntegerVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantIntegerVertex(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }

    @Test
    public void canTADFrom2D() {
        IntegerTADVertex rowOne = new IntegerTADVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new int[]{1, 2, 3}, rowOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{1, 3}, rowOne.getShape());

        IntegerTADVertex rowTwo = new IntegerTADVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new int[]{4, 5, 6}, rowTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{1, 3}, rowTwo.getShape());

        IntegerTADVertex columnOne = new IntegerTADVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new int[]{1, 4}, columnOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 1}, columnOne.getShape());

        IntegerTADVertex columnTwo = new IntegerTADVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new int[]{2, 5}, columnTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 1}, columnTwo.getShape());

        IntegerTADVertex columnThree = new IntegerTADVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new int[]{3, 6}, columnThree.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 1}, columnThree.getShape());
    }

    @Test
    public void canTadAlongACube() {
        IntegerVertex cube = new ConstantIntegerVertex(0);
        cube.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        IntegerTADVertex dimenZeroFace = new IntegerTADVertex(cube, 0, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 2}, dimenZeroFace.getShape());

        IntegerTADVertex dimenOneFace = new IntegerTADVertex(cube, 1, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 2}, dimenOneFace.getShape());

        IntegerTADVertex dimenTwoFace = new IntegerTADVertex(cube, 2, 0);
        Assert.assertArrayEquals(new int[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new int[]{2, 2}, dimenTwoFace.getShape());
    }

}
