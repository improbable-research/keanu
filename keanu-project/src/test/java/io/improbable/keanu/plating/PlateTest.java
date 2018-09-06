package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.when;

import java.util.Collection;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;

@RunWith(MockitoJUnitRunner.class)
public class PlateTest {

    public static final VertexLabel VERTEX_LABEL_1 = new VertexLabel("foo");
    private Plate plate;

    @Mock
    private Vertex<?> vertex;

    @Before
    public void createPlate() throws Exception {
        when(vertex.getLabel()).thenReturn(VERTEX_LABEL_1);
        plate = new Plate();
        plate.add(vertex);    }

    @Test
    public void youCanGetAVertexByName() {
        assertThat(plate.get(VERTEX_LABEL_1), equalTo(vertex));
    }

    @Test
    public void youCanGetAllTheProxyVertices() {
        DoubleProxyVertex proxy1 = new DoubleProxyVertex("proxy1");
        IntegerProxyVertex proxy2 = new IntegerProxyVertex("proxy2");
        BoolProxyVertex proxy3 = new BoolProxyVertex("proxy3");
        plate.add(proxy1);
        plate.add(proxy2);
        plate.add(proxy3);

        Collection<Vertex<?>> proxies = plate.getProxyVertices();
        assertThat(proxies, containsInAnyOrder(proxy1, proxy2, proxy3));
    }
}
