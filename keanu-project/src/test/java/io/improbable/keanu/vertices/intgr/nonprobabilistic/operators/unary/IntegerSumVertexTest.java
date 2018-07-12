package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class IntegerSumVertexTest {

    @Test
    public void doesSum() {

        IntegerVertex in = new UniformIntVertex(new int[]{1, 5}, 0, 10);
        in.setValue(new int[]{1, 2, 3, 4, 5});
        IntegerVertex summed = in.sum();

        assertEquals(1 + 2 + 3 + 4 + 5, summed.eval().scalar().intValue());
    }
}
