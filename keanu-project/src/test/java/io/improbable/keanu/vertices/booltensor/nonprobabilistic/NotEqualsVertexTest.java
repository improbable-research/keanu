package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.ConstantVertex;
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
        NotEqualsVertex<Tensor<T>, Tensor<T>> vertex = new NotEqualsVertex<>(
            new ConstantVertex<>(Tensor.scalar(a)),
            new ConstantVertex<>(Tensor.scalar(b))
        );
        assertEquals(expected, vertex.lazyEval().scalar());
    }
}
