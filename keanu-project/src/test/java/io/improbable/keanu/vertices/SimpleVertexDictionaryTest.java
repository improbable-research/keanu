package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.sameInstance;
import static org.mockito.Mockito.mock;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;

public class SimpleVertexDictionaryTest {

    Vertex vertex1 = mock(Vertex.class);
    Vertex vertex2 = mock(Vertex.class);
    VertexLabel label1 = new VertexLabel("label 1");
    VertexLabel label2 = new VertexLabel("label 2");

    Map<VertexLabel, Vertex<?>> map;
    SimpleVertexDictionary dictionary;

    @Before
    public void setUp() throws Exception {
        map = Maps.newHashMap();
        map.put(label1, vertex1);
        map.put(label2, vertex2);

        dictionary = SimpleVertexDictionary.backedBy(map);
    }

    @Test
    public void ifYouChangeTheUnderlyingMapItChangesTheDictionary() {
        VertexLabel label3 = new VertexLabel("label3");
        Vertex<?> vertex3 = mock(Vertex.class);
        map.put(label3, vertex3);
        assertThat(dictionary.get(label3), sameInstance(vertex3));
    }

    @Test
    public void youCanCombineTwoVertexDictionaries() {
        VertexLabel label3 = new VertexLabel("label3");
        Vertex<?> vertex3 = mock(Vertex.class);
        SimpleVertexDictionary dictionary2 = SimpleVertexDictionary.backedBy(ImmutableMap.of(label3, vertex3));

        VertexDictionary combinedDictionary = SimpleVertexDictionary.combine(dictionary, dictionary2);
        assertThat(combinedDictionary.get(label1), sameInstance(vertex1));
        assertThat(combinedDictionary.get(label2), sameInstance(vertex2));
        assertThat(combinedDictionary.get(label3), sameInstance(vertex3));
    }

    @Test
    public void youCanAddExtraEntriesAndGetANewDictionary() {
        VertexLabel label3 = new VertexLabel("label3");
        Vertex<?> vertex3 = mock(Vertex.class);
        VertexDictionary newDictionary = dictionary.withExtraEntries(ImmutableMap.of(label3, vertex3));

        assertThat(newDictionary.get(label1), sameInstance(vertex1));
        assertThat(newDictionary.get(label2), sameInstance(vertex2));
        assertThat(newDictionary.get(label3), sameInstance(vertex3));
    }
}
