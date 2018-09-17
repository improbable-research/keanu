package io.improbable.keanu.plating;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.when;

import java.util.Collection;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;

@RunWith(MockitoJUnitRunner.class)
public class PlateTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    public static final VertexLabel VERTEX_LABEL_1 = new VertexLabel("foo");
    private Plate plate;

    @Mock
    private Vertex<?> vertex;

    @Before
    public void createPlate() throws Exception {
        when(vertex.getLabel()).thenReturn(VERTEX_LABEL_1);
        plate = new Plate();
        plate.add(vertex);
    }

    @Test
    public void youCanGetAVertexByName() {
        Vertex<?> vertex = plate.get(VERTEX_LABEL_1);
        assertThat(vertex, equalTo(this.vertex));
    }

    @Test
    public void youCanGetAllTheProxyVertices() {
        DoubleProxyVertex proxy1 = new DoubleProxyVertex(new VertexLabel("proxy1"));
        IntegerProxyVertex proxy2 = new IntegerProxyVertex(new VertexLabel("proxy2"));
        BoolProxyVertex proxy3 = new BoolProxyVertex(new VertexLabel("proxy3"));
        plate.add(proxy1);
        plate.add(proxy2);
        plate.add(proxy3);

        Collection<Vertex<?>> proxies = plate.getProxyVertices();
        assertThat(proxies, containsInAnyOrder(proxy1, proxy2, proxy3));
    }

    @Test
    public void itThrowsIfYouAddAVertexWithNoLabel() {
        expectedException.expect(PlateException.class);
        expectedException.expectMessage(endsWith(" must contain a label in order to be added to a plate"));
        plate.add(ConstantVertex.of(1.));
    }
}
