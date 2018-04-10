package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
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
        LessThanOrEqualVertex<Integer, Integer> vertex = new LessThanOrEqualVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }

    private void isLessThanOrEqual(double a, double b, boolean expected) {
        LessThanOrEqualVertex<Double, Double> vertex = new LessThanOrEqualVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }
}
