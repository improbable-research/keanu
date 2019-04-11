from keanu.description_creator import create_description
from keanu.vertex import Const, OrBinary, AndBinary, DoubleIf, Binomial


def test_simple_if_vertex_description_created_correctly() -> None:
    predicate = Const(False)
    three = Const(3.0)
    four = Const(4.0)

    three.set_label("Three")
    four.set_label("Four")

    result = DoubleIf(predicate, three, four)
    assert create_description(result) == "This Vertex = Const(false) ? Three : Four"


def test_simple_binary_op_description() -> None:
    two = Const(2.0)
    three = Const(3.0)
    three.set_label("Three")

    pred1 = two >= three

    assert create_description(pred1) == "This Vertex = Const(2.0) >= Three"
