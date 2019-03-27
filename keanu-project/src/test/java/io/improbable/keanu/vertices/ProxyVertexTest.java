package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;
import org.junit.Test;

public class ProxyVertexTest {

    @Test(expected = RuntimeException.class)
    public void testCannotResetVertexLabel() {
        IntegerProxyVertex vertex = new IntegerProxyVertex(new VertexLabel("label1"));
        vertex.setLabel("two");
    }
}
