package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class SumVertexTest {

    @Test
    public void doesSum() {

        DoubleVertex in = new UniformVertex(new int[]{1, 5}, 0, 10);
        in.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex summed = in.sum();

        assertEquals(1 + 2 + 3 + 4 + 5, summed.lazyEval().scalar(), 1e-5);
        assertArrayEquals(
            new double[]{1, 1, 1, 1, 1},
            summed.getDualNumber().getPartialDerivatives().withRespectTo(in).asFlatDoubleArray(),
            1e-5
        );
    }
}
