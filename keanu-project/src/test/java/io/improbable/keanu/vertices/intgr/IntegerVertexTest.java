package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class IntegerVertexTest {

    IntegerVertex v1;
    IntegerVertex v2;

    @Before
    public void setup() {
        v1 = ConstantVertex.of(3);
        v2 = ConstantVertex.of(2);
    }

    @Test
    public void doesMultiply() {
        IntegerVertex result = v1.multiply(v2);
        result.lazyEval();
        Integer expected = 6;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesAdd() {
        IntegerVertex result = v1.plus(v2);
        result.lazyEval();
        Integer expected = 5;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesSubtract() {
        IntegerVertex result = v1.minus(v2);
        result.lazyEval();
        Integer expected = 1;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesObserve() {
        PoissonVertex testIntegerVertex = new PoissonVertex(1.0);
        testIntegerVertex.observe(5);

        Integer expected = 5;
        assertEquals(testIntegerVertex.getValue().scalar(), expected);
        assertTrue(testIntegerVertex.isObserved());
    }

    @Test
    public void doesLambda() {
        Function<IntegerTensor, IntegerTensor> op = val -> val.plus(5);

        IntegerVertex result = v1.lambda(op);
        result.lazyEval();
        Integer expected = 8;
        assertEquals(result.getValue().scalar(), expected);
    }

}
