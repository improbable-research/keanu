from keanu.vertex.label import _VertexLabel


def test_you_can_create_a_label_with_a_name() -> None:
    label = _VertexLabel("foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "foo"


def test_you_can_create_a_label_with_a_name_and_namespace() -> None:
    label = _VertexLabel("outer", "inner", "foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "outer.inner.foo"
