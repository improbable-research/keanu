package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.equalTo;

import java.util.Collection;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;

public class PlateTest {

    public static final String VERTEX_LABEL_1 = "foo";
    private Plate plate;
    private Vertex<?> vertex = null;

    @Before
    public void createPlate() throws Exception {
        plate = new Plate();
        plate.add(VERTEX_LABEL_1, vertex);    }

    @Test
    public void youCanGetAVertexByName() {
        assertThat(plate.get(VERTEX_LABEL_1), equalTo(vertex));
    }

    @Test
    public void youCanGetAllTheProxyVertices() {
        DoubleProxyVertex proxy1 = new DoubleProxyVertex();
        IntegerProxyVertex proxy2 = new IntegerProxyVertex();
        BoolProxyVertex proxy3 = new BoolProxyVertex();
        plate.add("proxy1", proxy1);
        plate.add("proxy2", proxy2);
        plate.add("proxy3", proxy3);

        Collection<Vertex<?>> proxies = plate.getProxyVertices();
        assertThat(proxies, containsInAnyOrder(proxy1, proxy2, proxy3));
    }
}
