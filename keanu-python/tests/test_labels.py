import pytest

from keanu.vertex.label import _VertexLabel


def test_you_can_create_a_label_with_a_name() -> None:
    label = _VertexLabel("foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "foo"


def test_you_can_create_a_label_with_a_name_and_namespace() -> None:
    label = _VertexLabel("outer", "inner", "foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "outer.inner.foo"


def test_you_can_create_a_label_from_a_dot_separated_string() -> None:
    label = _VertexLabel.create_with_namespace("outer.inner.foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "outer.inner.foo"


def test_theres_a_factory_for_when_you_dont_know_if_the_string_is_namespaced() -> None:
    label = _VertexLabel.create_maybe_with_namespace("outer.inner.foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "outer.inner.foo"
    label = _VertexLabel.create_maybe_with_namespace("foo")
    assert label.get_unqualified_name() == "foo"
    assert label.get_qualified_name() == "foo"


def test_it_throws_if_you_use_the_wrong_separator() -> None:
    with pytest.raises(ValueError, match='No namespace separator "." found in outer/inner/foo'):
        _VertexLabel.create_with_namespace("outer/inner/foo")
