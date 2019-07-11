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
        IntegerVertex rowOne = matrixA.slice(0, 0);

        Assert.assertArrayEquals(new int[]{1, 2, 3}, rowOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        IntegerVertex rowTwo = matrixA.slice(0, 1);

        Assert.assertArrayEquals(new int[]{4, 5, 6}, rowTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        IntegerVertex columnOne = matrixA.slice(1, 0);

        Assert.assertArrayEquals(new int[]{1, 4}, columnOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        IntegerVertex columnTwo = matrixA.slice(1, 1);

        Assert.assertArrayEquals(new int[]{2, 5}, columnTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        IntegerVertex columnThree = matrixA.slice(1, 2);

        Assert.assertArrayEquals(new int[]{3, 6}, columnThree.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3() {
        IntegerVertex cube = new ConstantIntegerVertex(0);
        cube.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        IntegerVertex dimenZeroFace = cube.slice(0, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        IntegerVertex dimenOneFace = cube.slice(1, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        IntegerVertex dimenTwoFace = cube.slice(2, 0);
        Assert.assertArrayEquals(new int[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

}
