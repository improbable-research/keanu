package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
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
        GreaterThanOrEqualVertex<Integer, Integer> vertex = new GreaterThanOrEqualVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }

    private void isGreaterThanOrEqual(double a, double b, boolean expected) {
        GreaterThanOrEqualVertex<Double, Double> vertex = new GreaterThanOrEqualVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }
}
