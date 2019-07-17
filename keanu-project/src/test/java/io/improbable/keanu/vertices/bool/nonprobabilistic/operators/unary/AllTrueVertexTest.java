package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;

public class AllTrueVertexTest {

    @Test
    public void canAllTrueWhenNotAllTrue() {
        BooleanVertex a = new ConstantBooleanVertex(true, false, true, false).reshape(2, 2);
        assertFalse(a.allTrue().getValue().scalar());
    }

    @Test
    public void canAllTrueWhenAllTrue() {
        BooleanVertex a = new ConstantBooleanVertex(true, true, true, true).reshape(2, 2);
        assertTrue(a.allTrue().getValue().scalar());
    }
}
