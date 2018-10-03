package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.SimpleBooleanTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import org.junit.Assert;
import org.junit.Test;

public class BoolConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        ConstantBoolVertex a = new ConstantBoolVertex(new boolean[] {true, true, true});
        ConstantBoolVertex b = new ConstantBoolVertex(new boolean[] {false, false, false});
        ConstantBoolVertex c = new ConstantBoolVertex(new boolean[] {true, true, true});

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, a, b);
        BoolConcatenationVertex concatOne = new BoolConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new int[] {2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new int[] {1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 0, 0, 0}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 0, 0, 0, 1, 1, 1},
                concatOne.getValue().asFlatDoubleArray(),
                0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        ConstantBoolVertex a = new ConstantBoolVertex(new boolean[] {true, true, true});
        ConstantBoolVertex b =
                new ConstantBoolVertex(new boolean[] {false, false, false, false, false, false});

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[] {1, 9}, concatZero.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 0, 0, 0, 0, 0, 0},
                concatZero.getValue().asFlatDoubleArray(),
                0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        ConstantBoolVertex a = new ConstantBoolVertex(new boolean[] {true, true, true});
        ConstantBoolVertex b = new ConstantBoolVertex(false);

        BoolConcatenationVertex concat = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[] {1, 4}, concat.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 0}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        ConstantBoolVertex a = new ConstantBoolVertex(false);
        ConstantBoolVertex b = new ConstantBoolVertex(new boolean[] {true, true, true});

        BoolConcatenationVertex concat = new BoolConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[] {1, 4}, concat.getShape());
        Assert.assertArrayEquals(
                new double[] {0, 1, 1, 1}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        ConstantBoolVertex a = new ConstantBoolVertex(new boolean[] {true, true, true});
        ConstantBoolVertex b = new ConstantBoolVertex(new boolean[] {false, false});

        new BoolConcatenationVertex(0, a, b);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        ConstantBoolVertex m =
                new ConstantBoolVertex(
                        new SimpleBooleanTensor(
                                new boolean[] {true, true, true, true}, new int[] {2, 2}));
        ConstantBoolVertex a =
                new ConstantBoolVertex(
                        new SimpleBooleanTensor(
                                new boolean[] {false, false, false, false}, new int[] {2, 2}));

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, m, a);

        Assert.assertArrayEquals(new int[] {4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 1, 0, 0, 0, 0},
                concatZero.getValue().asFlatDoubleArray(),
                0.001);

        BoolConcatenationVertex concatOne = new BoolConcatenationVertex(1, m, a);

        Assert.assertArrayEquals(new int[] {2, 4}, concatOne.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 0, 0, 1, 1, 0, 0},
                concatOne.getValue().asFlatDoubleArray(),
                0.001);
    }

    @Test
    public void canConcatHighDimensionalShapes() {
        ConstantBoolVertex a =
                new ConstantBoolVertex(
                        new SimpleBooleanTensor(
                                new boolean[] {true, true, true, true, true, true, true, true},
                                new int[] {2, 2, 2}));
        ConstantBoolVertex b =
                new ConstantBoolVertex(
                        new SimpleBooleanTensor(
                                new boolean[] {
                                    false, false, false, false, false, false, false, false
                                },
                                new int[] {2, 2, 2}));

        BoolConcatenationVertex concatZero = new BoolConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new int[] {4, 2, 2}, concatZero.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                concatZero.getValue().asFlatDoubleArray(),
                0.001);

        BoolConcatenationVertex concatThree = new BoolConcatenationVertex(2, a, b);

        Assert.assertArrayEquals(new int[] {2, 2, 4}, concatThree.getShape());
        Assert.assertArrayEquals(
                new double[] {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0},
                concatThree.getValue().asFlatDoubleArray(),
                0.001);
    }
}
