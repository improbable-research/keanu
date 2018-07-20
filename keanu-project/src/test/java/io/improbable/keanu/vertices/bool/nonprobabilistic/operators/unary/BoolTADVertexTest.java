package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

public class BoolTADVertexTest {

    private BoolVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, true, false, false, true, true}, 2, 3));
    }

    @Test
    public void canTADFrom2D() {
        System.out.println(Arrays.toString(matrixA.getShape()));
        BoolTADVertex rowOne = new BoolTADVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new double[]{1, 1, 0}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowOne.getShape());

        BoolTADVertex rowTwo = new BoolTADVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new double[]{0, 1, 1}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowTwo.getShape());

        BoolTADVertex columnOne = new BoolTADVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new double[]{1, 0}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnOne.getShape());

        BoolTADVertex columnTwo = new BoolTADVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new double[]{1, 1}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnTwo.getShape());

        BoolTADVertex columnThree = new BoolTADVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new double[]{0, 1}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnThree.getShape());
    }

    @Test
    public void canTadAlongACube() {
        BoolVertex cube = new ConstantBoolVertex(false);
        cube.setValue(BooleanTensor.create(new boolean[]{true, true, false, false, true, true, false, false}, 2, 2, 2));

        BoolTADVertex dimenZeroFace = new BoolTADVertex(cube, 0, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenZeroFace.getShape());

        BoolTADVertex dimenOneFace = new BoolTADVertex(cube, 1, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenOneFace.getShape());

        BoolTADVertex dimenTwoFace = new BoolTADVertex(cube, 2, 0);
        Assert.assertArrayEquals(new double[]{1, 0, 1, 0}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenTwoFace.getShape());
    }

}
