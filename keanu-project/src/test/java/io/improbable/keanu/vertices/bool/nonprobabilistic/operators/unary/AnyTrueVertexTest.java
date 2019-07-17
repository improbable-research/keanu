package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class AnyTrueVertexTest {

    @Test
    public void canAllTrueWhenNot() {
        BooleanVertex a = new ConstantBooleanVertex(false, false, false, false).reshape(2, 2);
        assertFalse(a.anyTrue().getValue().scalar());
    }

    @Test
    public void canAllTrueWhen() {
        BooleanVertex a = new ConstantBooleanVertex(false, false, true, false).reshape(2, 2);
        assertTrue(a.anyTrue().getValue().scalar());
    }
}
