package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class MatrixMultiplicationVertexTest {

    @Test
    public void canSimpleMatrixMultiply() {
        DoubleTensor matrixA = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        DoubleTensor matrixB = DoubleTensor.create(new double[]{2, 4, 6, 8}, 2, 2);

        MatrixMultiplicationVertex mmul = new MatrixMultiplicationVertex(ConstantVertex.of(matrixA), ConstantVertex.of(matrixB));

        DoubleTensor mmulResult = mmul.lazyEval();

        DoubleTensor expected = DoubleTensor.create(new double[]{14, 20, 30, 44}, 2, 2);

        assertEquals(expected, mmulResult);
    }

    @Test
    public void canDoMatrixMultiply2x2() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);

        //of c wrt a,b
        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dCdb = c.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleTensor expecteddCda = DoubleTensor.create(new double[]{
            5, 7,
            0, 0,
            6, 8,
            0, 0,
            0, 0,
            5, 7,
            0, 0,
            6, 8
        }, new int[]{2, 2, 2, 2});

        DoubleTensor expecteddCdb = DoubleTensor.create(new double[]{
            1, 0,
            2, 0,
            0, 1,
            0, 2,
            3, 0,
            4, 0,
            0, 3,
            0, 4
        }, new int[]{2, 2, 2, 2});

        //of d wrt a,b
        assertEquals(expecteddCda, dCda);
        assertEquals(expecteddCdb, dCdb);

        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dDda = d.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dDdb = d.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleTensor expecteddDda = DoubleTensor.create(new double[]{
            5, 0,
            6, 0,
            0, 5,
            0, 6,
            7, 0,
            8, 0,
            0, 7,
            0, 8
        }, new int[]{2, 2, 2, 2});

        DoubleTensor expecteddDdb = DoubleTensor.create(new double[]{
            1, 3,
            0, 0,
            2, 4,
            0, 0,
            0, 0,
            1, 3,
            0, 0,
            2, 4
        }, new int[]{2, 2, 2, 2});

        assertEquals(expecteddDda, dDda);
        assertEquals(expecteddDdb, dDdb);

        DoubleVertex e = c.plus(d);

        //of e wrt a, b
        DoubleTensor dEda = e.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dEdb = e.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleTensor expecteddEda = expecteddDda.plus(expecteddCda);
        DoubleTensor expecteddEdb = expecteddDdb.plus(expecteddCdb);

        assertEquals(expecteddEda, dEda);
        assertEquals(expecteddEdb, dEdb);
    }

    @Test
    public void canDoMatrixMultiplyAutoDiff() {

        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2}, 1, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{1, 3, 5, 2, 4, 6}, 2, 3));

        DoubleVertex N = m.matrixMultiply(alpha);
        DualNumber NDual = N.getDualNumber();

        DoubleTensor dNdm = NDual.getPartialDerivatives().withRespectTo(m);
        DoubleTensor expectedDNdm = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 1, 3, 1, 2);

        assertEquals(expectedDNdm, dNdm);

        DoubleTensor dNdAlpha = NDual.getPartialDerivatives().withRespectTo(alpha);
        DoubleTensor expectedDNdAlpha = DoubleTensor.create(new double[]{
            1, 0, 0,
            2, 0, 0,
            0, 1, 0,
            0, 2, 0,
            0, 0, 1,
            0, 0, 2
        }, 1, 3, 2, 3);

        assertEquals(expectedDNdAlpha, dNdAlpha);
    }

    @Test
    public void canDoDoubleMatrixMultiplyAutoDiff() {

        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{
            1, 2
        }, 1, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{
            1, 3,
            2, 4
        }, 2, 2));

        DoubleVertex beta = new UniformVertex(0, 10);
        beta.setValue(DoubleTensor.create(new double[]{
            5, 7,
            6, 8
        }, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);
        DoubleVertex y = N.matrixMultiply(beta);
        DualNumber yDual = y.getDualNumber();

        DoubleTensor dydm = yDual.getPartialDerivatives().withRespectTo(m);
        DoubleTensor expectedDydm = DoubleTensor.create(new double[]{
            23, 34, 31, 46
        }, 1, 2, 1, 2);

        assertEquals(expectedDydm, dydm);

        DoubleTensor dydalpha = yDual.getPartialDerivatives().withRespectTo(alpha);
        DoubleTensor expectedDydalpha = DoubleTensor.create(new double[]{
            5, 6,
            10, 12,
            7, 8,
            14, 16
        }, 1, 2, 2, 2);

        assertEquals(expectedDydalpha, dydalpha);

        DoubleTensor dydbeta = yDual.getPartialDerivatives().withRespectTo(beta);
        DoubleTensor expectedDydbeta = DoubleTensor.create(new double[]{
            5, 0,
            11, 0,
            0, 5,
            0, 11
        }, 1, 2, 2, 2);

        assertEquals(expectedDydbeta, dydbeta);
    }

    @Test
    public void canDoTripleMatrixMultiplyAutoDiff() {

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
        DualNumber yDual = y.getDualNumber();

        DoubleTensor dydalpha = yDual.getPartialDerivatives().withRespectTo(alpha);
        DoubleTensor expectedDydalpha = DoubleTensor.create(new double[]{
            56, 92,
            103, 174,
            66, 108,
            117, 198,
            76, 124,
            131, 222
        }, 3, 1, 2, 2);

        assertEquals(expectedDydalpha, dydalpha);
    }
}
