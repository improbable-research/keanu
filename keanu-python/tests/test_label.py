import pytest
from keanu.vertex import VertexLabel


@pytest.fixture
def label1() -> VertexLabel:
    return VertexLabel("label1", ["inner", "outer"])


def test_equality(label1: VertexLabel) -> None:
    label1_clone = VertexLabel("label1", ["inner", "outer"])
    assert label1 == label1_clone


def test_inequality(label1: VertexLabel) -> None:
    label2 = VertexLabel("label2", ["inner", "outer"])
    assert label1 != label2


def test_equality_str_with_no_namespace(label1: VertexLabel) -> None:
    label_with_no_namespace = VertexLabel("nonamespace")
    assert label_with_no_namespace == "nonamespace"


def test_inequality_str_with_no_namespace(label1: VertexLabel) -> None:
    label_with_no_namespace = VertexLabel("nonamespace")
    assert label_with_no_namespace != "withnamespace"


def test_inequality_str_with_namespace_separator(label1: VertexLabel) -> None:
    assert label1 != "outer.inner.label1"


def test_is_in_namespace(label1: VertexLabel) -> None:
    assert label1.is_in_namespace(["inner", "outer"])


def test_is_not_in_namespace(label1: VertexLabel) -> None:
    assert not label1.is_in_namespace(["outer", "inner"])


def test_with_extra_name_space(label1: VertexLabel) -> None:
    top_level = "top_level"
    expected = VertexLabel("label1", ["inner", "outer", top_level])
    assert label1.with_extra_namespace(top_level) == expected


def test_without_outer_namespace(label1: VertexLabel) -> None:
    expected = VertexLabel("label1", ["inner"])
    assert label1.without_outer_namespace() == expected


def test_get_outer_namespace(label1: VertexLabel) -> None:
    assert label1.get_outer_namespace() == "outer"


def test_get_none_outer_namespace() -> None:
    assert VertexLabel("label without namespace").get_outer_namespace() is None


def test_unqualified_name(label1: VertexLabel) -> None:
    assert label1.get_unqualified_name() == "label1"


def test_qualified_name(label1: VertexLabel) -> None:
    assert label1.get_qualified_name() == "outer.inner.label1"


def test_dict_distinguishes_label_namespaced_and_str_with_just_namespace_separator(label1: VertexLabel) -> None:
    similar_to_label1 = VertexLabel(label1.get_qualified_name())
    d = {}
    d[similar_to_label1] = 2
    d[label1] = 1

    assert len(d) == 2
    assert d[similar_to_label1] == 2
    assert d[label1] == 1
