package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Maps;

public class SimpleVertexDictionaryTest {

    Vertex vertex1 = mock(Vertex.class);
    Vertex vertex2 = mock(Vertex.class);
    VertexLabel label1 = new VertexLabel("label 1");
    VertexLabel label2 = new VertexLabel("label 2");

    Map<VertexLabel, Vertex<?>> map;
    VertexDictionary dictionary;

    @Before
    public void setUp() throws Exception {
        map = Maps.newHashMap();
        map.put(label1, vertex1);
        map.put(label2, vertex2);

        dictionary = SimpleVertexDictionary.backedBy(map);
    }

    @Test
    public void youCanGetAllTheVerticesAsANewDictionary() {
        VertexDictionary dictionary2 = dictionary.getAllVertices();
        assertThat(dictionary2.get(label1), sameInstance(vertex1));
        assertThat(dictionary2.get(label2), sameInstance(vertex2));
    }

    @Test
    public void ifYouChangeTheUnderlyingMapItChangesTheDictionary() {
        VertexLabel label3 = new VertexLabel("label3");
        Vertex<?> vertex3 = mock(Vertex.class);
        map.put(label3, vertex3);
        assertThat(dictionary.get(label3), sameInstance(vertex3));
    }

    @Test
    public void whenYouGetAllVerticesItsADefensiveCopy() {
        VertexDictionary dictionary2 = dictionary.getAllVertices();
        assertThat(dictionary2.get(label1), sameInstance(vertex1));
        assertThat(dictionary2.get(label2), sameInstance(vertex2));
        VertexLabel label3 = new VertexLabel("label3");
        map.put(label3, mock(Vertex.class));
        assertThat(dictionary2.get(label3), is(nullValue()));
    }
}
