package io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class AllFalseVertexTest {

    @Test
    public void canAllFalseWhenNot() {
        BooleanVertex a = new ConstantBooleanVertex(true, false, true, false).reshape(2, 2);
        assertFalse(a.allFalse().getValue().scalar());
    }

    @Test
    public void canAllFalseWhen() {
        BooleanVertex a = new ConstantBooleanVertex(false, false, false, false).reshape(2, 2);
        assertTrue(a.allFalse().getValue().scalar());
    }
}
