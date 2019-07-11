package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TakeVertexTest {

    @Test
    public void takeFromVectorCorrectlyTakesPartialToo() {
        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 1, 4));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 1, 4));

        DoubleVertex N = m.multiply(alpha);

        DoubleVertex take = N.take(0, 0);
        DoubleTensor takePartial = Differentiator.forwardModeAutoDiff(m, take).of(take);
        DoubleTensor takePartialReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(m);

        assertEquals(N.getValue(0, 0), take.getValue().scalar(), 1e-6);
        assertArrayEquals(new long[]{1, 4}, takePartial.getShape());
        assertArrayEquals(new double[]{10, 0, 0, 0}, takePartial.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(takePartial.getShape(), takePartialReverse.getShape());
        assertArrayEquals(takePartial.asFlatDoubleArray(), takePartialReverse.asFlatDoubleArray(), 1e-6);

        DoubleVertex take2 = N.take(0, 1);
        DoubleTensor takePartial2 = Differentiator.forwardModeAutoDiff(m, take2).of(take2);
        DoubleTensor takePartial2Reverse = Differentiator.reverseModeAutoDiff(take2, m, alpha).withRespectTo(m);

        assertEquals(N.getValue(0, 1), take2.getValue().scalar(), 1e-6);
        assertArrayEquals(new long[]{1, 4}, takePartial2.getShape());
        assertArrayEquals(new double[]{0, 15, 0, 0}, takePartial2.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(takePartial2.getShape(), takePartial2Reverse.getShape());
        assertArrayEquals(takePartial2.asFlatDoubleArray(), takePartial2Reverse.asFlatDoubleArray(), 1e-6);
    }


    @Test
    public void takeFromMatrixCorrectlyTakesPartialToo() {
        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        DoubleVertex take = N.take(0, 0);

        DoubleTensor takePartial = Differentiator.forwardModeAutoDiff(m, take).of(take);
        DoubleTensor takePartialReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(m);

        assertArrayEquals(new long[]{2, 2}, takePartial.getShape());
        assertArrayEquals(new double[]{10, 20, 0, 0}, takePartial.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(takePartial.getShape(), takePartialReverse.getShape());
        assertArrayEquals(takePartial.asFlatDoubleArray(), takePartialReverse.asFlatDoubleArray(), 1e-6);

        DoubleVertex take2 = N.take(0, 1);

        DoubleTensor takePartial2 = Differentiator.forwardModeAutoDiff(m, take2).of(take2);
        DoubleTensor takePartial2Reverse = Differentiator.reverseModeAutoDiff(take2, m, alpha).withRespectTo(m);

        assertArrayEquals(new long[]{2, 2}, takePartial2.getShape());
        assertArrayEquals(new double[]{15, 25, 0, 0}, takePartial2.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(takePartial.getShape(), takePartial2Reverse.getShape());
        assertArrayEquals(takePartial2.asFlatDoubleArray(), takePartial2Reverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void takeFromMatrixMultiplyCorrectlyTakesPartialToo() {

        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{
            1,
            2
        }, 2, 1));

        UniformVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{
            1, 3,
            2, 4
        }, 2, 2));

        UniformVertex beta = new UniformVertex(0, 10);
        beta.setValue(DoubleTensor.create(new double[]{
            5, 8,
            6, 9,
            7, 10
        }, 3, 2));

        DoubleVertex N = alpha.matrixMultiply(m);
        DoubleVertex L = beta.matrixMultiply(alpha);
        //y = L x N = (beta x alpha) x (alpha x m)
        DoubleVertex y = L.matrixMultiply(N);

        DoubleVertex take = y.take(0, 0);
        DoubleTensor takeDiff = Differentiator.forwardModeAutoDiff(alpha, take).of(take);
        DoubleTensor takeDiffReverse = Differentiator.reverseModeAutoDiff(take, m, alpha).withRespectTo(alpha);

        assertArrayEquals(new long[]{2, 2}, takeDiff.getShape());
        assertArrayEquals(new double[]{56, 92, 103, 174}, takeDiff.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(takeDiff.getShape(), takeDiffReverse.getShape());
        assertArrayEquals(takeDiff.asFlatDoubleArray(), takeDiffReverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void shapeOfPartialCorrectlyPassesThroughTake() {
        UniformVertex A = new UniformVertex(0, 10);
        A.setValue(DoubleTensor.arange(1, 28).reshape(3, 3, 3));

        UniformVertex B = new UniformVertex(0, 10);
        B.setValue(DoubleTensor.arange(1, 28).reshape(3, 3, 3));

        DoubleVertex C = A.times(B);

        DoubleVertex D = C.take(0, 1, 2);

        DoubleVertex E = new UniformVertex(0, 10);
        E.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 1, 4));

        DoubleVertex F = D.plus(E);

        DoubleTensor dFWrtAReverse = Differentiator.reverseModeAutoDiff(F, A, B).withRespectTo(A);
        DoubleTensor dFWrtAForward = Differentiator.forwardModeAutoDiff(A, F).of(F);

        assertArrayEquals(new long[]{1, 4, 3, 3, 3}, dFWrtAForward.getShape());
        assertArrayEquals(dFWrtAForward.getShape(), dFWrtAReverse.getShape());
        assertArrayEquals(dFWrtAForward.asFlatDoubleArray(), dFWrtAReverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{3, 3, 3}, -10.0, 10.0);
        UniformVertex inputB = new UniformVertex(new long[]{3, 3, 3}, -10.0, 10.0);
        UniformVertex inputC = new UniformVertex(new long[]{2, 2}, -10.0, 10.0);
        DoubleVertex outputVertex = inputA.times(10.0).times(inputB).take(0, 1, 2).plus(inputC);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA, inputB), outputVertex, 10.0, 1e-10);
    }

}
