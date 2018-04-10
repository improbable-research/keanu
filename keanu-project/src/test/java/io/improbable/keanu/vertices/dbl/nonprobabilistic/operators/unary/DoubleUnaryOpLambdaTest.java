package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DoubleUnaryOpLambdaTest {

    @Test
    public void GIVEN_a_double_vertex_THEN_transform() {

        DoubleVertex CONST = new ConstantDoubleVertex(2.0);
        DoubleVertex v1 = new DoubleUnaryOpLambda<>(CONST, (val) -> val * 2);

        assertEquals(4.0, v1.lazyEval(), 0.001);
    }

}
