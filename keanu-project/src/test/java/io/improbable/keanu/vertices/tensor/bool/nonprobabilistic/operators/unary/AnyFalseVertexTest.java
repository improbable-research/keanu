package io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class AnyFalseVertexTest {

    @Test
    public void canAnyFalseWhenNot() {
        BooleanVertex a = new ConstantBooleanVertex(true, true, true, true).reshape(2, 2);
        assertFalse(a.anyFalse().getValue().scalar());
    }

    @Test
    public void canAnyFalseWhen() {
        BooleanVertex a = new ConstantBooleanVertex(false, false, true, false).reshape(2, 2);
        assertTrue(a.anyFalse().getValue().scalar());
    }
}
