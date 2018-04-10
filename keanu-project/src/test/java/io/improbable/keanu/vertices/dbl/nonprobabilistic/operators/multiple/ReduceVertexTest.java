package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ReduceVertexTest {
    int minValue = -8;
    int maxValue = 12;
    int total = 0;

    List<Vertex<Double>> verts = new LinkedList<>();

    @Before
    public void prepare() {
        for (int i = minValue; i <= maxValue; i++) {
            DoubleVertex v = new ConstantDoubleVertex((double) i);
            verts.add(v);
            total += i;
        }
    }

    @Test
    public void calculatesSumCorrectly() {
        DoubleVertex sum = new DoubleReduceVertex(verts, (a, b) -> (a + b));
        assertEquals(sum.lazyEval(), total, 0.0001);
    }

    @Test
    public void calculatesMaxCorrectly() {
        DoubleVertex max = new DoubleReduceVertex(verts, Math::max);
        assertEquals(max.lazyEval(), maxValue, 0.0001);
    }

    @Test
    public void calculatesMinCorrectly() {
        DoubleVertex min = new DoubleReduceVertex(verts, Math::min);
        assertEquals(min.lazyEval(), minValue, 0.0001);
    }

    @Test
    public void varargsConstrution() {
        DoubleVertex max = new DoubleReduceVertex(Math::max, null, verts.get(0), verts.get(1));
        assertEquals(max.lazyEval(), Math.max(verts.get(0).lazyEval(), verts.get(1).lazyEval()), 0.0001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void zeroArgThrowsExcpetion() {
        DoubleVertex min = new DoubleReduceVertex(Math::max, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void singleArgThrowsExcpetion() {
        DoubleVertex min = new DoubleReduceVertex(Math::max, null, verts.get(0));
    }

    @Test
    public void doubleArgExecutesAsExpected() {
        DoubleVertex min = new DoubleReduceVertex(Math::max, null, verts.get(0), verts.get(1));
        assertEquals(min.lazyEval(), Math.max(verts.get(0).lazyEval(), verts.get(1).lazyEval()), 0.0);
    }
}
