package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GreaterThanVertexTest {

    @Test
    public void comparesIntegers() {
        isGreaterThan(0, 1, false);
        isGreaterThan(1, 1, false);
        isGreaterThan(2, 1, true);
    }

    @Test
    public void comparesDoubles() {
        isGreaterThan(0.0, 0.5, false);
        isGreaterThan(0.5, 0.5, false);
        isGreaterThan(1.0, 0.5, true);
    }

    private void isGreaterThan(int a, int b, boolean expected) {
        GreaterThanVertex<Integer, Integer> vertex = new GreaterThanVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }

    private void isGreaterThan(double a, double b, boolean expected) {
        GreaterThanVertex<Double, Double> vertex = new GreaterThanVertex<>(new ConstantVertex<>(a), new ConstantVertex<>(b));
        assertEquals(expected, vertex.lazyEval());
    }

}
