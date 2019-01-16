package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Assert;
import org.junit.Test;

public class BooleanConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, true}, 1, 3));
        ConstantBooleanVertex b = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{false, false, false}, 1, 3));
        ConstantBooleanVertex c = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, true}, 1, 3));

        BooleanConcatenationVertex concatZero = new BooleanConcatenationVertex(0, a, b);
        BooleanConcatenationVertex concatOne = new BooleanConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new long[]{2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new long[]{1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0, 1, 1, 1}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{false, false, false, false, false, false});

        BooleanConcatenationVertex concatZero = new BooleanConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{9}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{true, true, true});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{false});

        BooleanConcatenationVertex concat = new BooleanConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 0}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(new boolean[]{false});
        ConstantBooleanVertex b = new ConstantBooleanVertex(new boolean[]{true, true, true});

        BooleanConcatenationVertex concat = new BooleanConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{0, 1, 1, 1}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, true}, 1, 3));
        ConstantBooleanVertex b = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{false, false}, 1, 2));

        new BooleanConcatenationVertex(0, a, b);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        ConstantBooleanVertex m = new ConstantBooleanVertex(new SimpleBooleanTensor(new boolean[]{true, true, true, true}, new long[]{2, 2}));
        ConstantBooleanVertex a = new ConstantBooleanVertex(new SimpleBooleanTensor(new boolean[]{false, false, false, false}, new long[]{2, 2}));

        BooleanConcatenationVertex concatZero = new BooleanConcatenationVertex(0, m, a);

        Assert.assertArrayEquals(new long[]{4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1, 0, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);

        BooleanConcatenationVertex concatOne = new BooleanConcatenationVertex(1, m, a);

        Assert.assertArrayEquals(new long[]{2, 4}, concatOne.getShape());
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0, 1, 1, 0, 0}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatHighDimensionalShapes() {
        ConstantBooleanVertex a = new ConstantBooleanVertex(
            new SimpleBooleanTensor(new boolean[]{true, true, true, true, true, true, true, true}, new long[]{2, 2, 2}));
        ConstantBooleanVertex b = new ConstantBooleanVertex(
            new SimpleBooleanTensor(new boolean[]{false, false, false, false, false, false, false, false}, new long[]{2, 2, 2}));

        BooleanConcatenationVertex concatZero = new BooleanConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4, 2, 2}, concatZero.getShape());
        Assert.assertArrayEquals(
            new double[]{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
            concatZero.getValue().asFlatDoubleArray(),
            0.001
        );

        BooleanConcatenationVertex concatThree = new BooleanConcatenationVertex(2, a, b);

        Assert.assertArrayEquals(new long[]{2, 2, 4}, concatThree.getShape());
        Assert.assertArrayEquals(
            new double[]{1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
            concatThree.getValue().asFlatDoubleArray(),
            0.001
        );
    }

}
