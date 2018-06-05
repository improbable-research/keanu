package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TensorReduceVertexTest {
    int minValue = -8;
    int maxValue = 12;
    int total = 0;

    List<Vertex<DoubleTensor>> verts = new LinkedList<>();

    @Before
    public void prepare() {
        for (int i = minValue; i <= maxValue; i++) {
            DoubleTensorVertex v = new ConstantDoubleTensorVertex((double) i);
            verts.add(v);
            total += i;
        }
    }

    @Test
    public void calculatesSumCorrectly() {
        DoubleTensorVertex sum = new TensorDoubleReduceVertex(verts, (a, b) -> (a.plus(b)));
        assertEquals(sum.lazyEval().scalar(), total, 0.0001);
    }

    @Test
    public void calculatesMaxCorrectly() {
        DoubleTensorVertex max = new TensorDoubleReduceVertex(verts, DoubleTensor::max);
        assertEquals(max.lazyEval().scalar(), maxValue, 0.0001);
    }

    @Test
    public void calculatesMinCorrectly() {
        DoubleTensorVertex min = new TensorDoubleReduceVertex(verts, DoubleTensor::min);
        assertEquals(min.lazyEval().scalar(), minValue, 0.0001);
    }

    @Test
    public void varargsConstrution() {
        DoubleTensorVertex max = new TensorDoubleReduceVertex(DoubleTensor::max, null, verts.get(0), verts.get(1));
        assertEquals(max.lazyEval().scalar(), Math.max(verts.get(0).lazyEval().scalar(), verts.get(1).lazyEval().scalar()), 0.0001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void zeroArgThrowsException() {
        DoubleTensorVertex min = new TensorDoubleReduceVertex(DoubleTensor::max, null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void singleArgThrowsException() {
        DoubleTensorVertex min = new TensorDoubleReduceVertex(DoubleTensor::max, null, verts.get(0));
    }

    @Test
    public void doubleArgExecutesAsExpected() {
        DoubleTensorVertex min = new TensorDoubleReduceVertex(DoubleTensor::max, null, verts.get(0), verts.get(1));
        assertEquals(min.lazyEval().scalar(), Math.max(verts.get(0).lazyEval().scalar(), verts.get(1).lazyEval().scalar()), 0.0);
    }
}
