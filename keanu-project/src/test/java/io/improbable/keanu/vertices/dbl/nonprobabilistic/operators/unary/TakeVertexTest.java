package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

public class TakeVertexTest {

    @Test
    public void takeFromVectorCorrectlyTakesPartialToo() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 1, 4));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 1, 4));

        DoubleVertex N = m.multiply(alpha);

        TakeVertex take = new TakeVertex(N, 0, 0);
        DoubleTensor takePartial = take.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor takePartialReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(m);

        Assert.assertEquals(N.getValue(0, 0), take.getValue().scalar(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, takePartial.getShape());
        Assert.assertArrayEquals(new double[]{10, 0, 0, 0}, takePartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, takePartialReverse.getShape());
        Assert.assertArrayEquals(new double[]{10, 0, 0, 0}, takePartialReverse.asFlatDoubleArray(), 1e-6);

        TakeVertex take2 = new TakeVertex(N, 0, 1);
        DoubleTensor takePartial2 = take2.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor takePartial2Reverse = Differentiator.reverseModeAutoDiff(take2, m, alpha).withRespectTo(m);

        Assert.assertEquals(N.getValue(0, 1), take2.getValue().scalar(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, takePartial2.getShape());
        Assert.assertArrayEquals(new double[]{0, 15, 0, 0}, takePartial2.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new int[]{1, 1, 1, 4}, takePartial2Reverse.getShape());
        Assert.assertArrayEquals(new double[]{0, 15, 0, 0}, takePartial2Reverse.asFlatDoubleArray(), 1e-6);
    }


    @Test
    public void takeFromMatrixCorrectlyTakesPartialToo() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        TakeVertex take = new TakeVertex(N, 0, 0);

        DoubleTensor takePartial = take.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor takePartialReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(m);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, takePartial.getShape());
        Assert.assertArrayEquals(new double[]{10, 20, 0, 0}, takePartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(takePartial.getShape(), takePartialReverse.getShape());
        Assert.assertArrayEquals(takePartial.asFlatDoubleArray(), takePartialReverse.asFlatDoubleArray(), 1e-6);

        TakeVertex take2 = new TakeVertex(N, 0, 1);

        DoubleTensor takePartial2 = take2.getDualNumber().getPartialDerivatives().withRespectTo(m);
        DoubleTensor takePartial2Reverse = Differentiator.reverseModeAutoDiff(take2, m, alpha).withRespectTo(m);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, takePartial2.getShape());
        Assert.assertArrayEquals(new double[]{15, 25, 0, 0}, takePartial2.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(takePartial.getShape(), takePartial2Reverse.getShape());
        Assert.assertArrayEquals(takePartial2.asFlatDoubleArray(), takePartial2Reverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void takeFromMatrixMultiplyCorrectlyTakesPartialToo() {

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

        TakeVertex take = new TakeVertex(y, 0, 0);
        DoubleTensor takeDual = take.getDualNumber().getPartialDerivatives().withRespectTo(alpha);
        DoubleTensor takeDualReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(alpha);

        Assert.assertArrayEquals(new int[]{1, 1, 2, 2}, takeDual.getShape());
        Assert.assertArrayEquals(new double[]{56, 92, 103, 174}, takeDual.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(takeDual.getShape(), takeDualReverse.getShape());
        Assert.assertArrayEquals(takeDual.asFlatDoubleArray(), takeDualReverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void shapeOfPartialCorrectlyPassesThroughTake() {
        DoubleVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex C = A.times(B);

        DoubleVertex D = C.take(0, 1);

        DoubleVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 1, 4));

        DoubleVertex F = D.plus(E);

        PartialDerivatives forward = F.getDualNumber().getPartialDerivatives();
        PartialDerivatives reverse = Differentiator.reverseModeAutoDiff(F, A, B);

        Assert.assertArrayEquals(new int[]{1, 4, 2, 2}, forward.withRespectTo(A).getShape());
        Assert.assertArrayEquals(forward.withRespectTo(A).getShape(), reverse.withRespectTo(A).getShape());
        Assert.assertArrayEquals(forward.withRespectTo(A).asFlatDoubleArray(), reverse.withRespectTo(A).asFlatDoubleArray(), 1e-6);
    }

}
