package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex;
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
        LessThanVertex<IntegerTensor, IntegerTensor> vertex = new LessThanVertex<>(new ConstantIntegerVertex(a), new ConstantIntegerVertex(b));
        assertEquals(expected, vertex.lazyEval().scalar());
    }

    private void isLessThan(double a, double b, boolean expected) {
        LessThanVertex<DoubleTensor, DoubleTensor> vertex = new LessThanVertex<>(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
        assertEquals(expected, vertex.lazyEval().scalar());
    }
}
