import pytest

from keanu.vertex.label import _VertexLabel


def test_you_can_create_a_label_with_a_name() -> None:
    label = _VertexLabel("foo")
    assert label.unwrap().getUnqualifiedName() == "foo"
    assert label.get_name() == "foo"


def test_you_can_create_a_label_from_a_dot_separated_string() -> None:
    label = _VertexLabel("outer.inner.foo")
    assert label.unwrap().getUnqualifiedName() == "foo"
    assert label.get_name() == "outer.inner.foo"


def test_there_is_a_helper_method_to_build_the_string_from_a_list() -> None:
    label = _VertexLabel.create_from_list("outer", "inner", "foo")
    assert label.unwrap().getUnqualifiedName() == "foo"
    assert label.get_name() == "outer.inner.foo"


def test_you_cannot_build_it_with_an_empty_list() -> None:
    with pytest.raises(ValueError, match="You must pass in at least one string"):
        _VertexLabel.create_from_list()


def test_the_repr_method_makes_it_clear_what_the_namespace_is() -> None:
    label = _VertexLabel("outer.inner.foo")
    assert str(label) == "['outer', 'inner', 'foo']"
