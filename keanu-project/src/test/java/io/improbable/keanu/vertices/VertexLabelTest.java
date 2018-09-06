package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;

import org.junit.Test;

public class VertexLabelTest {
    @Test
    public void byDefaultALabelHasNoNamespace() {
        VertexLabel foo = new VertexLabel("foo");
        assertThat(foo.toString(), equalTo("null:foo"));
    }

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

    @Test
    public void vertexLabelsWithTheSameNameButDifferentNamespacesAreNotEqual() {
        VertexLabel foo1 = new VertexLabel("namespace1", "foo");
        VertexLabel foo2 = new VertexLabel("namespace2", "foo");
        assertThat(foo1, not(equalTo(foo2)));
        assertThat(foo1.hashCode(), not(equalTo(foo2.hashCode())));
    }

    @Test
    public void aVertexLabelsWithANamespaceIsNoteEqualToOneWithoutANamespace() {
        VertexLabel foo1 = new VertexLabel("namespace", "foo");
        VertexLabel foo2 = new VertexLabel("foo");
        assertThat(foo1, not(equalTo(foo2)));
        assertThat(foo1.hashCode(), not(equalTo(foo2.hashCode())));
    }

    @Test
    public void vertexLabelsWithTheSameNamespaceDifferentNamesAreNotEqual() {
        VertexLabel foo = new VertexLabel("namespace", "foo");
        VertexLabel bar = new VertexLabel("namespace", "bar");
        assertThat(foo, not(equalTo(bar)));
        assertThat(foo.hashCode(), not(equalTo(bar.hashCode())));
    }

    @Test
    public void youCanSetTheNamespace() {
        String namespace = "namespace";
        String name = "foo";
        VertexLabel foo = new VertexLabel(name);
        VertexLabel newFoo = foo.inNamespace(namespace);
        assertThat(newFoo, equalTo(new VertexLabel(namespace, name)));
    }
}
