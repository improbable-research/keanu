package io.improbable.keanu.vertices.bool.nonprobabilistic;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;

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
        BooleanVertex vertex = ConstantVertex.of(a).equalTo(ConstantVertex.of(b));
        assertEquals(expected, vertex.eval().scalar());
    }
}
