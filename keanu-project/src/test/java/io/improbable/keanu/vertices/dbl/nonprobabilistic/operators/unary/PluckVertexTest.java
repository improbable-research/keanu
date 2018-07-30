package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

public class PluckVertexTest {

    @Test
    public void canPluckPartialDerivativeFromVector() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 1, 4));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 1, 4));

        DoubleVertex N = m.multiply(alpha);

        PluckVertex pluck = new PluckVertex(N, 0, 0);
        DoubleTensor pluckedPartial = pluck.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertEquals(N.getValue(0, 0), pluck.getValue().scalar(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, pluckedPartial.getShape());
        Assert.assertArrayEquals(new double[]{10, 0, 0, 0}, pluckedPartial.asFlatDoubleArray(), 1e-6);

        PluckVertex pluck2 = new PluckVertex(N, 0, 1);
        DoubleTensor pluckedPartial2 = pluck2.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertEquals(N.getValue(0, 1), pluck2.getValue().scalar(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, pluckedPartial2.getShape());
        Assert.assertArrayEquals(new double[]{0, 15, 0, 0}, pluckedPartial2.asFlatDoubleArray(), 1e-6);
    }


    @Test
    public void canPluckPartialDerivativeFromMatrix() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        PluckVertex pluck = new PluckVertex(N, 0, 0);

        DoubleTensor pluckedPartial = pluck.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, pluckedPartial.getShape());
        Assert.assertArrayEquals(new double[]{10, 20, 0, 0}, pluckedPartial.asFlatDoubleArray(), 1e-6);

        PluckVertex pluck2 = new PluckVertex(N, 0, 1);

        DoubleTensor pluckedPartial2 = pluck2.getDualNumber().getPartialDerivatives().withRespectTo(m);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, pluckedPartial2.getShape());
        Assert.assertArrayEquals(new double[]{0, 0, 10, 20}, pluckedPartial2.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canPluckFromComplexMatrixMul() {

        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{
            1,
            2
        }, 2, 1));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{
            1, 3,
            2, 4
        }, 2, 2));

        DoubleVertex beta = new UniformVertex(0, 10);
        beta.setValue(DoubleTensor.create(new double[]{
            5, 8,
            6, 9,
            7, 10
        }, 3, 2));

        DoubleVertex N = alpha.matrixMultiply(m);
        DoubleVertex L = beta.matrixMultiply(alpha);
        //y = L x N = (beta x alpha) x (alpha x m)
        DoubleVertex y = L.matrixMultiply(N);

        PluckVertex pluck = new PluckVertex(y, 0, 0);
        DoubleTensor pluckedDual = pluck.getDualNumber().getPartialDerivatives().withRespectTo(alpha);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, pluckedDual.getShape());
        Assert.assertArrayEquals(new double[]{56, 92, 103, 174}, pluckedDual.asFlatDoubleArray(), 1e-6);
    }

}
