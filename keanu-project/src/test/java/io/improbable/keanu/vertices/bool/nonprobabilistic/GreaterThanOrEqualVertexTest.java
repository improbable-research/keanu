package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GreaterThanOrEqualVertexTest {

    @Test
    public void comparesIntegers() {
        isGreaterThanOrEqual(0, 1, false);
        isGreaterThanOrEqual(1, 1, true);
        isGreaterThanOrEqual(2, 1, true);
    }

    @Test
    public void comparesDoubles() {
        isGreaterThanOrEqual(0.0, 0.5, false);
        isGreaterThanOrEqual(0.5, 0.5, true);
        isGreaterThanOrEqual(1.0, 0.5, true);
    }

    private void isGreaterThanOrEqual(int a, int b, boolean expected) {
        GreaterThanOrEqualVertex<IntegerTensor, IntegerTensor> vertex = new GreaterThanOrEqualVertex<>(new ConstantIntegerVertex(a), new ConstantIntegerVertex(b));
        assertEquals(expected, vertex.lazyEval().scalar());
    }

    private void isGreaterThanOrEqual(double a, double b, boolean expected) {
        GreaterThanOrEqualVertex<DoubleTensor, DoubleTensor> vertex = new GreaterThanOrEqualVertex<>(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
        assertEquals(expected, vertex.lazyEval().scalar());
    }
}
