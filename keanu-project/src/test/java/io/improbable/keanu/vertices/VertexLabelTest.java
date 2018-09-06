package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;

import org.junit.Test;

public class VertexLabelTest {
    @Test
    public void vertexLabelsWithTheSameNameAreEqual() {
        VertexLabel foo1 = new VertexLabel("foo");
        VertexLabel foo2 = new VertexLabel("foo");
        assertThat(foo1, equalTo(foo2));
        assertThat(foo1.hashCode(), equalTo(foo2.hashCode()));
    }

    @Test
    public void vertexLabelsWithDifferentNamesAreNotEqual() {
        VertexLabel foo = new VertexLabel("foo");
        VertexLabel bar = new VertexLabel("bar");
        assertThat(foo, not(equalTo(bar)));
        assertThat(foo.hashCode(), not(equalTo(bar.hashCode())));
    }
}
