package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

public class ConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6});

        UniformVertex a2 = new UniformVertex(0.0, 1.0);
        a2.setValue(new double[]{7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(0, a, a1);
        ConcatenationVertex concatAlongOne = new ConcatenationVertex(1, a, a1, a2);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongOne.getValue().asFlatDoubleArray(), 0.001);

        Assert.assertArrayEquals(new int[]{2, 3}, concatAlongZero.getShape());
        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongOne.getShape());
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(1, a, a1);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);

        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongZero.getShape());
    }

    @Test (expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatAlongZero = new ConcatenationVertex(0, a, a1);

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatAlongZero.getValue().asFlatDoubleArray(), 0.001);

        Assert.assertArrayEquals(new int[]{1, 9}, concatAlongZero.getShape());
    }



}
