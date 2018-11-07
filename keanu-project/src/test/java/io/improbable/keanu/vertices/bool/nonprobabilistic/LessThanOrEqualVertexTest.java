package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.ConstantVertexFactory;
import io.improbable.keanu.vertices.bool.BoolVertex;
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
        BoolVertex vertex = ConstantVertexFactory.of(a).lessThanOrEqualTo(ConstantVertexFactory.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }

    private void isLessThanOrEqual(double a, double b, boolean expected) {
        BoolVertex vertex = ConstantVertexFactory.of(a).lessThanOrEqualTo(ConstantVertexFactory.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
