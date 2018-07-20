package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TADVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class TADVertexTest {

    private DoubleVertex matrixA;

    @Before
    public void setup() {
       matrixA = new ConstantDoubleVertex(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }

    @Test
    public void canTADFrom2D() {
        TADVertex rowOne = new TADVertex(matrixA, 0, 0);

        Assert.assertArrayEquals(new double[]{1, 2, 3}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowOne.getShape());

        TADVertex rowTwo = new TADVertex(matrixA, 0, 1);

        Assert.assertArrayEquals(new double[]{4, 5, 6}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 3}, rowTwo.getShape());

        TADVertex columnOne = new TADVertex(matrixA, 1, 0);

        Assert.assertArrayEquals(new double[]{1, 4}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnOne.getShape());

        TADVertex columnTwo = new TADVertex(matrixA, 1, 1);

        Assert.assertArrayEquals(new double[]{2, 5}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnTwo.getShape());

        TADVertex columnThree = new TADVertex(matrixA, 1, 2);

        Assert.assertArrayEquals(new double[]{3, 6}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, columnThree.getShape());
    }

    @Test
    public void canRepeatablyTADForAPick() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        TADVertex columnZero = new TADVertex(m, 0, 0);
        TADVertex elementZero = new TADVertex(columnZero, 0, 0);

        Assert.assertEquals(elementZero.getValue().scalar(), 1, 1e-6);
    }

    @Test
    public void splitCorrectlySplitsRowOfPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        TADVertex splitN = new TADVertex(N, 0, 0);

        DoubleTensor originalPartial = N.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor splitPartial = splitN.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(splitN.getValue().asFlatDoubleArray(), new double[]{50, 65}, 1e-6);
        Assert.assertArrayEquals(new int[]{1, 2}, splitN.getShape());

        Assert.assertArrayEquals(originalPartial.tad(0, 0).asFlatDoubleArray(), splitPartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 2, 2, 2}, splitPartial.getShape());
    }

    @Test
    public void splitCorrectlySplitsColumnOfPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        TADVertex splitN = new TADVertex(N, 1, 1);

        DoubleTensor originalPartial = N.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor splitPartial = splitN.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(splitN.getValue().asFlatDoubleArray(), new double[]{65, 145}, 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1}, splitN.getShape());

        Assert.assertArrayEquals(originalPartial.tad(1, 1).asFlatDoubleArray(), splitPartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 1, 2, 2}, splitPartial.getShape());
    }

    @Test
    public void canTadAlongACube() {
        DoubleVertex cube = new UniformVertex(0, 10);
        cube.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 2, 2}));

        TADVertex dimenZeroFace = new TADVertex(cube, 0, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenZeroFace.getShape());

        TADVertex dimenOneFace = new TADVertex(cube, 1, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenOneFace.getShape());

        TADVertex dimenTwoFace = new TADVertex(cube, 2, 0);
        Assert.assertArrayEquals(new double[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{2, 2}, dimenTwoFace.getShape());
    }

}
