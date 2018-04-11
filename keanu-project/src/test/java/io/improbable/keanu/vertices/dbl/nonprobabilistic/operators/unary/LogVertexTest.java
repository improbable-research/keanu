package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LogVertexTest {

    @Test
    public void calculatesLogCorrectly() {
        DoubleVertex A = new ConstantDoubleVertex(5.0);
        DoubleVertex logA = new LogVertex(A);

        assertEquals(logA.lazyEval(), Math.log(5), 0.0001);
    }
}
