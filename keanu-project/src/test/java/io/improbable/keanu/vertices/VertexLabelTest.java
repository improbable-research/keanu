package io.improbable.keanu.vertices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;

import org.junit.Test;

public class VertexLabelTest {
    @Test
    public void byDefaultALabelHasNoNamespace() {
        VertexLabel foo = new VertexLabel("foo");
        assertThat(foo.toString(), equalTo("foo"));
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
    public void vertexLabelsWithDifferentDepthNamespacesAreNotEqual() {
        VertexLabel foo1 = new VertexLabel("foo", "inner", "outer");
        VertexLabel foo2 = new VertexLabel("foo", "outer.inner");
        assertThat(foo1, not(equalTo(foo2)));
        assertThat(foo1.hashCode(), not(equalTo(foo2.hashCode())));
        assertThat(foo1.toString(), equalTo(foo2.toString()));
    }

    @Test
    public void youCanSetTheNamespace() {
        String namespace = "namespace";
        String name = "foo";
        VertexLabel foo = new VertexLabel(name);
        VertexLabel newFoo = foo.withExtraNamespace(namespace);
        assertThat(newFoo, equalTo(new VertexLabel(name, namespace)));
    }

    @Test
    public void youCanAugmentTheNamespace() {
        String innerNamespace = "inner";
        String outerNamespace = "outer";
        String name = "foo";
        VertexLabel foo = new VertexLabel(name, innerNamespace);
        VertexLabel newFoo = foo.withExtraNamespace(outerNamespace);
        assertThat(newFoo, equalTo(new VertexLabel(name, innerNamespace, outerNamespace)));
    }

    @Test
    public void itUsesADotToPrint() {
        VertexLabel foo = new VertexLabel("foo", "inner", "outer");
        assertThat(foo.toString(), equalTo("outer.inner.foo"));
    }

    @Test
    public void youCanDiminishTheNamespace() {
        String innerNamespace = "inner";
        String outerNamespace = "outer";
        String name = "foo";
        VertexLabel foo = new VertexLabel(name, innerNamespace, outerNamespace);
        VertexLabel newFoo = foo.withoutOuterNamespace();
        assertThat(newFoo, equalTo(new VertexLabel(name, innerNamespace)));
    }

    @Test(expected = VertexLabelException.class)
    public void itThrowsIfYouDiminishTheNamespaceButThereIsNone() {
        VertexLabel foo = new VertexLabel("foo");
        foo.withoutOuterNamespace();
    }
}
