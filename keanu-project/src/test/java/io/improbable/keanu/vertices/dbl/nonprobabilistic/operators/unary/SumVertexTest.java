package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class SumVertexTest {

    @Test
    public void doesSum() {

        DoubleVertex in = new UniformVertex(new int[]{1, 5}, 0, 10);
        in.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex summed = in.sum();

        DoubleTensor wrtIn = summed.getDualNumber().getPartialDerivatives().withRespectTo(in);
        DoubleTensor expectedWrtIn = DoubleTensor.ones(1, 1, 1, 5);

        assertEquals(1 + 2 + 3 + 4 + 5, summed.lazyEval().scalar(), 1e-5);
        assertArrayEquals(
            expectedWrtIn.asFlatDoubleArray(),
            wrtIn.asFlatDoubleArray(),
            1e-5
        );
        assertArrayEquals(expectedWrtIn.getShape(), wrtIn.getShape());
    }

    @Test
    public void doesSumSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(new int[]{1, 5}, 0, 10);
        a.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex b = a.sum();

        DoubleTensor dbdaForward = b.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dbdaReverse = Differentiator.reverseModeAutoDiff(b, a).withRespectTo(a);

        DoubleTensor expectedDbDa = DoubleTensor.create(new double[]{1, 1, 1, 1, 1}, 1, 1, 1, 5);

        assertEquals(expectedDbDa, dbdaForward);
        assertEquals(expectedDbDa, dbdaReverse);
    }

    @Test
    public void canDoSumAutoDiffWhenSumIsNotWrtOrOf() {
        DoubleVertex a = new UniformVertex(new int[]{2, 3}, 0, 10);
        a.setValue(DoubleTensor.arange(0, 6).reshape(2, 3));

        DoubleVertex d = a.sum();

        DoubleVertex e = new UniformVertex(new int[]{2, 2}, 0, 10);
        e.setValue(DoubleTensor.arange(4, 8).reshape(2, 2));

        DoubleVertex f = d.times(e);

        DoubleTensor dfdaForward = f.getDualNumber().getPartialDerivatives().withRespectTo(a);

        PartialDerivatives dfdx = Differentiator.reverseModeAutoDiff(f, a, e);
        DoubleTensor dfdaReverse = dfdx.withRespectTo(a);

        DoubleTensor expectedDfdx = DoubleTensor.create(new double[]{
            4, 4, 4,
            4, 4, 4,
            5, 5, 5,
            5, 5, 5,
            6, 6, 6,
            6, 6, 6,
            7, 7, 7,
            7, 7, 7
        }, 2, 2, 2, 3);

        assertEquals(dfdaReverse, expectedDfdx);
        assertEquals(dfdaForward, expectedDfdx);
    }
}
