package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

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
        BooleanVertex vertex = ConstantVertex.of(a).lessThanOrEqualTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }

    private void isLessThanOrEqual(double a, double b, boolean expected) {
        BooleanVertex vertex = ConstantVertex.of(a).lessThanOrEqualTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
