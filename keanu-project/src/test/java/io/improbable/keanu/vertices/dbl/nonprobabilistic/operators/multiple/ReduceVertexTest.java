package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ReduceVertexTest {
    int minValue = -8;
    int maxValue = 12;
    int total = 0;

    List<Vertex<DoubleTensor>> verts = new LinkedList<>();

    @Before
    public void prepare() {
        for (double i = minValue; i <= maxValue; i++) {
            DoubleVertex v = ConstantVertex.of(i);
            verts.add(v);
            total += i;
        }
    }

    @Test
    public void calculatesSumCorrectly() {
        DoubleVertex sum = new DoubleReduceVertex(verts, (a, b) -> (a.plus(b)));
        assertEquals(sum.eval().scalar(), total, 0.0001);
    }

    @Test
    public void calculatesMaxCorrectly() {
        DoubleVertex max = new DoubleReduceVertex(verts, DoubleTensor::max);
        assertEquals(max.eval().scalar(), maxValue, 0.0001);
    }

    @Test
    public void calculatesMinCorrectly() {
        DoubleVertex min = new DoubleReduceVertex(verts, DoubleTensor::min);
        assertEquals(min.eval().scalar(), minValue, 0.0001);
    }

    @Test
    public void varargsConstrution() {
        DoubleVertex max = new DoubleReduceVertex(DoubleTensor::max, null, verts.get(0), verts.get(1));
        assertEquals(max.eval().scalar(), Math.max(verts.get(0).eval().scalar(), verts.get(1).eval().scalar()), 0.0001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void zeroArgThrowsException() {
        DoubleVertex min = new DoubleReduceVertex(DoubleTensor::max, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void singleArgThrowsException() {
        DoubleVertex min = new DoubleReduceVertex(DoubleTensor::max, null, verts.get(0));
    }

    @Test
    public void doubleArgExecutesAsExpected() {
        DoubleVertex min = new DoubleReduceVertex(DoubleTensor::max, null, verts.get(0), verts.get(1));
        assertEquals(min.eval().scalar(), Math.max(verts.get(0).eval().scalar(), verts.get(1).eval().scalar()), 0.0);
    }
}
