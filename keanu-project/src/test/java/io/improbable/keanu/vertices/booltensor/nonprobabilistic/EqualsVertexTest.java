package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.generic.SimpleTensor;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class EqualsVertexTest {

    @Test
    public void comparesIntegers() {
        equals(1, 1, true);
        equals(1, 0, false);
    }

    @Test
    public void comparesDoubles() {
        equals(1.0, 1.0, true);
        equals(1.0, 0.0, false);
    }

    @Test
    public void comparesObjects() {
        Object obj = new Object();
        equals(obj, obj, true);

        equals("test", "otherTest", false);
    }

    private <T> void equals(T a, T b, boolean expected) {
        EqualsVertex<SimpleTensor<T>> vertex = new EqualsVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.lazyEval().scalar());
    }
}
