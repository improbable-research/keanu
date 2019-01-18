package io.improbable.keanu.plating;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.Collection;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class PlateTest {

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    public static final VertexLabel VERTEX_LABEL_1 = new VertexLabel("foo");
    public static final VertexLabel VERTEX_LABEL_2 = new VertexLabel("bar");
    private Plate plate;

    @Mock
    private Vertex<?> vertex1;

    @Mock
    private Vertex<?> vertex2;

    @Before
    public void createPlate() throws Exception {
        when(vertex1.getLabel()).thenReturn(VERTEX_LABEL_1);
        plate = new Plate();
        plate.add(vertex1);
        plate.add(VERTEX_LABEL_2, vertex2);
    }

    @Test
    public void youCanGetAVertexByName() {
        Vertex<?> vertex = plate.get(VERTEX_LABEL_1);
        assertThat(vertex, equalTo(this.vertex1));
    }

    @Test
    public void unlabelledVerticesCanAlsoBeGotIfYouKnowTheLabelToUse() {
        Vertex<?> vertex = plate.get(VERTEX_LABEL_2);
        assertThat(vertex, equalTo(this.vertex2));
    }

    @Test
    public void youCanGetAllTheProxyVertices() {
        DoubleProxyVertex proxy1 = new DoubleProxyVertex(new VertexLabel("proxy1"));
        IntegerProxyVertex proxy2 = new IntegerProxyVertex(new VertexLabel("proxy2"));
        BooleanProxyVertex proxy3 = new BooleanProxyVertex(new VertexLabel("proxy3"));
        plate.add(proxy1);
        plate.add(proxy2);
        plate.add(proxy3);

        Collection<Vertex<?>> proxies = plate.getProxyVertices();
        assertThat(proxies, containsInAnyOrder(proxy1, proxy2, proxy3));
    }

    @Test
    public void itThrowsIfYouAddAVertexWithNoLabel() {
        expectedException.expect(PlateConstructionException.class);
        expectedException.expectMessage(endsWith(" must contain a label in order to be added to a plate"));
        plate.add(ConstantVertex.of(1.));
    }
}
