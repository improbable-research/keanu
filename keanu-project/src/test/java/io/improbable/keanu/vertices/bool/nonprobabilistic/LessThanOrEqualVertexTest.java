package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LessThanOrEqualVertexTest {

    @Test
    public void comparesIntegers() {
        isLessThanOrEqual(0, 1, true);
        isLessThanOrEqual(1, 1, true);
        isLessThanOrEqual(2, 1, false);
    }

    @Test
    public void comparesDoubles() {
        isLessThanOrEqual(0.0, 0.5, true);
        isLessThanOrEqual(0.5, 0.5, true);
        isLessThanOrEqual(1.0, 0.5, false);
    }

    private void isLessThanOrEqual(int a, int b, boolean expected) {
        LessThanOrEqualVertex<IntegerTensor, IntegerTensor> vertex = new LessThanOrEqualVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }

    private void isLessThanOrEqual(double a, double b, boolean expected) {
        LessThanOrEqualVertex<DoubleTensor, DoubleTensor> vertex = new LessThanOrEqualVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
