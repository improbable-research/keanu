from keanu.description_creator import DescriptionCreator
from keanu.vertex import Const, OrBinary, AndBinary, DoubleIf


def test_simple_descriptions_created_correctly() -> None:
    two = Const(2.0)
    three = Const(3.0)
    four = Const(4.0)

    three.set_label("Three")
    four.set_label("Four")

    result = (two * three) - four
    assert DescriptionCreator.create_description(result) == "This Vertex = (Const(2.0) * Three) - Four"


def test_integer_addition_and_multiplication_descriptions_created_correctly() -> None:
    two = Const(2)
    three = Const(3)
    four = Const(4)

    three.set_label("Three")
    four.set_label("Four")

    result = (two * three) + four
    assert DescriptionCreator.create_description(result) == "This Vertex = (Const(2) * Three) + Four"


def test_if_vertex_description_created_correctly() -> None:
    predicate = Const(False)
    three = Const(3.0)
    four = Const(4.0)

    three.set_label("Three")
    four.set_label("Four")

    result = DoubleIf(predicate, three, four)
    assert DescriptionCreator.create_description(result) == "This Vertex = Const(false) ? Three : Four"


def test_boolean_unary_ops_descriptions_created_correctly() -> None:
    two = Const(2.0)
    three = Const(3.0)
    false = Const(False)

    three.set_label("Three")

    pred1 = two >= three
    pred2 = two > three
    pred3 = two <= three
    pred4 = two < three

    pred5 = OrBinary(false, false)
    pred6 = AndBinary(false, false)

    assert DescriptionCreator.create_description(pred1) == "This Vertex = Const(2.0) >= Three"
    assert DescriptionCreator.create_description(pred2) == "This Vertex = Const(2.0) > Three"
    assert DescriptionCreator.create_description(pred3) == "This Vertex = Const(2.0) <= Three"
    assert DescriptionCreator.create_description(pred4) == "This Vertex = Const(2.0) < Three"
    assert DescriptionCreator.create_description(pred5) == "This Vertex = Const(false) || Const(false)"
    assert DescriptionCreator.create_description(pred6) == "This Vertex = Const(false) && Const(false)"
