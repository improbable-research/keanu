package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AlongDimensionVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class AlongDimensionVertexTest {

    private DoubleVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantDoubleVertex(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }

    @Test
    public void canTADFrom2D() {
        AlongDimensionVertex rowOne = new AlongDimensionVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new double[]{1, 2, 3}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowOne.getShape());

        AlongDimensionVertex rowTwo = new AlongDimensionVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new double[]{4, 5, 6}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowTwo.getShape());

        AlongDimensionVertex columnOne = new AlongDimensionVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new double[]{1, 4}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnOne.getShape());

        AlongDimensionVertex columnTwo = new AlongDimensionVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new double[]{2, 5}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnTwo.getShape());

        AlongDimensionVertex columnThree = new AlongDimensionVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new double[]{3, 6}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnThree.getShape());
    }

    @Test
    public void canRepeatablyTADForAPick() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        AlongDimensionVertex columnZero = new AlongDimensionVertex(m, 0, 0);
        AlongDimensionVertex elementZero = new AlongDimensionVertex(columnZero, 0, 0);

        Assert.assertEquals(elementZero.getValue().scalar(), 1, 1e-6);
    }

    @Test
    public void splitCorrectlySplitsRowOfPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        AlongDimensionVertex splitN = new AlongDimensionVertex(N, 0, 0);

        DoubleTensor originalPartial = N.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor splitPartial = splitN.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(splitN.getValue().asFlatDoubleArray(), new double[]{50, 65}, 1e-6);
        Assert.assertArrayEquals(new int[]{1, 2}, splitN.getShape());

        Assert.assertArrayEquals(originalPartial.alongDimension(0, 0).asFlatDoubleArray(), splitPartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 2, 2, 2}, splitPartial.getShape());
    }

    @Test
    public void splitCorrectlySplitsColumnOfPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        AlongDimensionVertex splitN = new AlongDimensionVertex(N, 1, 1);

        DoubleTensor originalPartial = N.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor splitPartial = splitN.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(splitN.getValue().asFlatDoubleArray(), new double[]{65, 145}, 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, splitN.getShape());

        Assert.assertArrayEquals(originalPartial.alongDimension(1, 1).asFlatDoubleArray(), splitPartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1, 2, 2}, splitPartial.getShape());
    }

    @Test
    public void canTadAlongACube() {
        DoubleVertex cube = new UniformVertex(0, 10);
        cube.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2}));

        AlongDimensionVertex dimenZeroFace = new AlongDimensionVertex(cube, 0, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenZeroFace.getShape());

        AlongDimensionVertex dimenOneFace = new AlongDimensionVertex(cube, 1, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenOneFace.getShape());

        AlongDimensionVertex dimenTwoFace = new AlongDimensionVertex(cube, 2, 0);
        Assert.assertArrayEquals(new double[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenTwoFace.getShape());
    }

}
