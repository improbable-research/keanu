package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

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
        NotEqualsVertex<GenericTensor<T>, GenericTensor<T>> vertex = new NotEqualsVertex<>(
            ConstantVertex.of(a),
            ConstantVertex.of(b)
        );
        assertEquals(expected, vertex.lazyEval().scalar());
    }
}
