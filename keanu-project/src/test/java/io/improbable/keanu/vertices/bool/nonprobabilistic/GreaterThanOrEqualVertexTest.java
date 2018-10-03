package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static org.junit.Assert.assertEquals;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import org.junit.Test;

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
        BoolVertex vertex = ConstantVertex.of(a).greaterThanOrEqualTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }

    private void isGreaterThanOrEqual(double a, double b, boolean expected) {
        BoolVertex vertex = ConstantVertex.of(a).greaterThanOrEqualTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
