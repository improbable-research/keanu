package io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.ConstantVertex;
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
        EqualsVertex<GenericTensor<T>> vertex = new EqualsVertex<>(ConstantVertex.of(a), ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
