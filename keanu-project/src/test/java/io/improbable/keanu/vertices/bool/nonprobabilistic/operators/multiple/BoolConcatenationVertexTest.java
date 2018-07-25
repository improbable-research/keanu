package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;

public class BoolConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{false, false, false});
        ConstantBooleanVertex c = new ConstantBooleanVertex(new boolean[]{true, true, true});

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, a, b);
        BoolConcatenationVertex concatOne = new BoolConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new int[]{2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new int[]{1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0, 1, 1, 1}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{false, false, false, false, false, false});

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 9}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(false);

        BoolConcatenationVertex concat = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(false);
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{true, true, true});

        BoolConcatenationVertex concat = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{0, 1, 1, 1}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test (expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{false, false});

        new BoolConcatenationVertex(0, a, b);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        ConstantBooleanVertex m = new ConstantBooleanVertex(new SimpleBooleanTensor(new boolean[]{true, true, true, true}, new int[]{2, 2}));
        ConstantBooleanVertex a = new ConstantBooleanVertex(new SimpleBooleanTensor(new boolean[]{false, false, false, false}, new int[]{2, 2}));

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, m, a);

        Assert.assertArrayEquals(new int[]{4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1, 0, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);

        BoolConcatenationVertex concatOne = new BoolConcatenationVertex(1, m, a);

        Assert.assertArrayEquals(new int[]{2, 4}, concatOne.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0, 1, 1, 0, 0}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatHighDimensionalShapes() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(
            new SimpleBooleanTensor(new boolean[]{true, true, true, true, true, true, true, true}, new int[]{2, 2, 2}));
        ConstantBooleanVertex b = new ConstantBooleanVertex(
            new SimpleBooleanTensor(new boolean[]{false, false, false, false, false, false, false, false}, new int[]{2, 2, 2}));

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new int[]{4, 2, 2}, concatZero.getShape());
        Assert.assertArrayEquals(
            new double[]{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            concatZero.getValue().asFlatDoubleArray(),
            0.001
        );

        BoolConcatenationVertex concatThree = new BoolConcatenationVertex(2, a, b);

        Assert.assertArrayEquals(new int[]{2, 2, 4}, concatThree.getShape());
        Assert.assertArrayEquals(
            new double[]{1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
            concatThree.getValue().asFlatDoubleArray(),
            0.001
        );
    }

}
