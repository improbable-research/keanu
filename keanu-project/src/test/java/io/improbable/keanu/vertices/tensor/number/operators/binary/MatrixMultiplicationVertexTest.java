package io.improbable.keanu.vertices.tensor.number.operators.binary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertEquals;

public class MatrixMultiplicationVertexTest {

    @Test
    public void canSimpleMatrixMultiply() {
        DoubleTensor matrixA = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        DoubleTensor matrixB = DoubleTensor.create(new double[]{2, 4, 6, 8}, 2, 2);

        DoubleVertex mmul = ConstantVertex.of(matrixA).matrixMultiply(ConstantVertex.of(matrixB));

        DoubleTensor mmulResult = mmul.lazyEval();

        DoubleTensor expected = DoubleTensor.create(new double[]{14, 20, 30, 44}, 2, 2);

        assertEquals(expected, mmulResult);
    }

    @Test
    public void canDoMatrixMultiply2x2() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);

        //of c wrt a,b
        DoubleTensor dCda = Differentiator.forwardModeAutoDiff(a, c).of(c);
        DoubleTensor dCdb = Differentiator.forwardModeAutoDiff(b, c).of(c);

        PartialsOf dCdxReverse = Differentiator.reverseModeAutoDiff(c, new HashSet<>(Arrays.asList(a, b)));
        DoubleTensor dCdaReverse = dCdxReverse.withRespectTo(a);
        DoubleTensor dCdbReverse = dCdxReverse.withRespectTo(b);

        DoubleTensor expecteddCda = DoubleTensor.create(new double[]{
            5, 7,
            0, 0,
            6, 8,
            0, 0,
            0, 0,
            5, 7,
            0, 0,
            6, 8
        }, new long[]{2, 2, 2, 2});

        DoubleTensor expecteddCdb = DoubleTensor.create(new double[]{
            1, 0,
            2, 0,
            0, 1,
            0, 2,
            3, 0,
            4, 0,
            0, 3,
            0, 4
        }, new long[]{2, 2, 2, 2});

        //of d wrt a,b
        assertEquals(expecteddCda, dCda);
        assertEquals(expecteddCdb, dCdb);
        assertEquals(expecteddCda, dCdaReverse);
        assertEquals(expecteddCdb, dCdbReverse);

        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dDda = Differentiator.forwardModeAutoDiff(a, d).of(d);
        DoubleTensor dDdb = Differentiator.forwardModeAutoDiff(b, d).of(d);

        PartialsOf dDdxReverse = Differentiator.reverseModeAutoDiff(d, new HashSet<>(Arrays.asList(a, b)));
        DoubleTensor dDdaReverse = dDdxReverse.withRespectTo(a);
        DoubleTensor dDdbReverse = dDdxReverse.withRespectTo(b);

        DoubleTensor expecteddDda = DoubleTensor.create(new double[]{
            5, 0,
            6, 0,
            0, 5,
            0, 6,
            7, 0,
            8, 0,
            0, 7,
            0, 8
        }, new long[]{2, 2, 2, 2});

        DoubleTensor expecteddDdb = DoubleTensor.create(new double[]{
            1, 3,
            0, 0,
            2, 4,
            0, 0,
            0, 0,
            1, 3,
            0, 0,
            2, 4
        }, new long[]{2, 2, 2, 2});

        assertEquals(expecteddDda, dDda);
        assertEquals(expecteddDdb, dDdb);
        assertEquals(expecteddDda, dDdaReverse);
        assertEquals(expecteddDdb, dDdbReverse);

        DoubleVertex e = c.plus(d);

        //of e wrt a, b
        DoubleTensor dEda = Differentiator.forwardModeAutoDiff(a, e).of(e);
        DoubleTensor dEdb = Differentiator.forwardModeAutoDiff(b, e).of(e);

        PartialsOf dEdxReverse = Differentiator.reverseModeAutoDiff(e, new HashSet<>(Arrays.asList(a, b)));
        DoubleTensor dEdaReverse = dEdxReverse.withRespectTo(a);
        DoubleTensor dEdbReverse = dEdxReverse.withRespectTo(b);

        DoubleTensor expecteddEda = expecteddDda.plus(expecteddCda);
        DoubleTensor expecteddEdb = expecteddDdb.plus(expecteddCdb);

        assertEquals(expecteddEda, dEda);
        assertEquals(expecteddEdb, dEdb);
        assertEquals(expecteddEda, dEdaReverse);
        assertEquals(expecteddEdb, dEdbReverse);
    }

    @Test
    public void canDoMatrixMultiplyAutoDiff() {

        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2}, 1, 2));

        UniformVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{1, 3, 5, 2, 4, 6}, 2, 3));

        DoubleVertex N = m.matrixMultiply(alpha);

        PartialsOf reverseModePartialDiff = Differentiator.reverseModeAutoDiff(N, m, alpha);

        DoubleTensor dNdmForward = Differentiator.forwardModeAutoDiff(m, N).of(N);
        DoubleTensor dNdmReverse = reverseModePartialDiff.withRespectTo(m);
        DoubleTensor expectedDNdm = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 1, 3, 1, 2);

        assertEquals(expectedDNdm, dNdmForward);
        assertEquals(expectedDNdm, dNdmReverse);

        DoubleTensor dNdAlphaForward = Differentiator.forwardModeAutoDiff(alpha, N).of(N);
        DoubleTensor dNdAlphaReverse = reverseModePartialDiff.withRespectTo(alpha);
        DoubleTensor expectedDNdAlpha = DoubleTensor.create(new double[]{
            1, 0, 0,
            2, 0, 0,
            0, 1, 0,
            0, 2, 0,
            0, 0, 1,
            0, 0, 2
        }, 1, 3, 2, 3);

        assertEquals(expectedDNdAlpha, dNdAlphaForward);
        assertEquals(expectedDNdAlpha, dNdAlphaReverse);
    }

    @Test
    public void canDoDoubleMatrixMultiplyAutoDiff() {

        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{
            1, 2
        }, 1, 2));

        UniformVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{
            1, 3,
            2, 4
        }, 2, 2));

        UniformVertex beta = new UniformVertex(0, 10);
        beta.setValue(DoubleTensor.create(new double[]{
            5, 7,
            6, 8
        }, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);
        DoubleVertex y = N.matrixMultiply(beta);

        PartialsOf dydx = Differentiator.reverseModeAutoDiff(y, m, alpha, beta);

        DoubleTensor dydmForward = Differentiator.forwardModeAutoDiff(m, y).of(y);
        DoubleTensor dydmReverse = dydx.withRespectTo(m);
        DoubleTensor expectedDydm = DoubleTensor.create(new double[]{
            23, 34, 31, 46
        }, 1, 2, 1, 2);

        assertEquals(expectedDydm, dydmForward);
        assertEquals(expectedDydm, dydmReverse);

        DoubleTensor dydalphaForward = Differentiator.forwardModeAutoDiff(alpha, y).of(y);
        DoubleTensor dydalphaReverse = dydx.withRespectTo(alpha);
        DoubleTensor expectedDydalpha = DoubleTensor.create(new double[]{
            5, 6,
            10, 12,
            7, 8,
            14, 16
        }, 1, 2, 2, 2);

        assertEquals(expectedDydalpha, dydalphaForward);
        assertEquals(expectedDydalpha, dydalphaReverse);

        DoubleTensor dydbetaForward = Differentiator.forwardModeAutoDiff(beta, y).of(y);
        DoubleTensor dydbetaReverse = dydx.withRespectTo(beta);
        DoubleTensor expectedDydbeta = DoubleTensor.create(new double[]{
            5, 0,
            11, 0,
            0, 5,
            0, 11
        }, 1, 2, 2, 2);

        assertEquals(expectedDydbeta, dydbetaForward);
        assertEquals(expectedDydbeta, dydbetaReverse);
    }

    @Test
    public void canDoTripleMatrixMultiplyAutoDiff() {

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
        PartialsOf dydx = Differentiator.reverseModeAutoDiff(y, alpha);

        DoubleTensor dydalphaForward = Differentiator.forwardModeAutoDiff(alpha, y).of(y);
        DoubleTensor dydalphaReverse = dydx.withRespectTo(alpha);
        DoubleTensor expectedDydalpha = DoubleTensor.create(new double[]{
            56, 92,
            103, 174,
            66, 108,
            117, 198,
            76, 124,
            131, 222
        }, 3, 1, 2, 2);

        assertEquals(expectedDydalpha, dydalphaForward);
        assertEquals(expectedDydalpha, dydalphaReverse);
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 5}, -10.0, 10.0);
        UniformVertex inputB = new UniformVertex(new long[]{5, 2}, -10.0, 10.0);
        DoubleVertex mmultVertex = inputA.matrixMultiply(inputB);
        DoubleVertex outputVertex = mmultVertex.times(
            new ConstantDoubleVertex(new double[]{1., 2., 3., 4., 5., 6., 7., 8.}, new long[]{2, 2, 2})
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA, inputB), outputVertex, INCREMENT, DELTA);
    }

    @Test
    public void changesMatchGradientWhenResultIsLengthOne() {
        UniformVertex inputA = new UniformVertex(new long[]{1, 2}, -10.0, 10.0);
        UniformVertex inputB = new UniformVertex(new long[]{2, 1}, -10.0, 10.0);
        DoubleVertex mmultVertex = inputA.matrixMultiply(inputB);

        DoubleVertex outputVertex = mmultVertex.times(
            new ConstantDoubleVertex(new double[]{1., 2., 3., 4., 5., 6., 7., 8.}, new long[]{2, 2, 2})
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA, inputB), outputVertex, INCREMENT, DELTA);
    }

}
