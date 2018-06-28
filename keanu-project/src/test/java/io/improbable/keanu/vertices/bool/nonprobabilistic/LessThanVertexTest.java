package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LessThanVertexTest {

    @Test
    public void comparesIntegers() {
        isLessThan(0, 1, true);
        isLessThan(1, 1, false);
        isLessThan(2, 1, false);
    }

    @Test
    public void comparesDoubles() {
        isLessThan(0.0, 0.5, true);
        isLessThan(0.5, 0.5, false);
        isLessThan(1.0, 0.5, false);
    }

    private void isLessThan(int a, int b, boolean expected) {
        LessThanVertex<IntegerTensor, IntegerTensor> vertex = new LessThanVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }

    private void isLessThan(double a, double b, boolean expected) {
        LessThanVertex<DoubleTensor, DoubleTensor> vertex = new LessThanVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
