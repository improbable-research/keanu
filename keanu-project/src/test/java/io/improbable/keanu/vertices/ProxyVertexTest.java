package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.IntegerProxyVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class ProxyVertexTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @Test
    public void testCannotResetVertexLabel() {
        expectedException.expect(RuntimeException.class);
        expectedException.expectMessage("You should not change the label on a Proxy Vertex");
        IntegerProxyVertex vertex = new IntegerProxyVertex(new VertexLabel("label1"));
        vertex.setLabel("two");
    }
}
