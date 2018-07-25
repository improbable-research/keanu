package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class NotEqualsVertexTest {

    @Test
    public void comparesIntegers() {
        equals(1, 1, false);
        equals(1, 0, true);
    }

    @Test
    public void comparesDoubles() {
        equals(1.0, 1.0, false);
        equals(1.0, 0.0, true);
    }

    @Test
    public void comparesObjects() {
        Object obj = new Object();
        equals(obj, obj, false);

        equals("test", "otherTest", true);
    }

    private <T> void equals(T a, T b, boolean expected) {
        BooleanVertex vertex = ConstantVertex.of(a).notEqualTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
