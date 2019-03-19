from keanu.vertex import Const, IntegerIf


def test_simple_descriptions_created_correctly() -> None:
    two = Const(2.0)
    three = Const(3.0)
    four = Const(4.0)

    three.set_label("Three")
    four.set_label("Four")

    result = (two * three) - four
    assert result.create_description() == "This Vertex = (Const(2.0) * Three) - Four"


def test_integer_addition_and_multiplication_descriptions_created_correctly() -> None:
    two = Const(2)
    three = Const(3)
    four = Const(4)

    three.set_label("Three")
    four.set_label("Four")

    result = (two * three) - four
    assert result.create_description() == "This Vertex = (Const(2) * Three) + Four"


def test_if_vertex_description_created_correctly() -> None:
    predicate = Const(False)
    three = Const(3.0)
    four = Const(4.0)

    three.set_label("Three")
    four.set_label("Four")

    result = IntegerIf(predicate, three, four)
    assert result.create_description() == "This Vertex = Const(false) ? Three : Four"


def test_boolean_unary_ops_descriptions_created_correctly() -> None:
    two = Const(2.0)
    three = Const(3.0)
    false = Const(False)

    three.set_label("Three")

    pred1 = two > three
    pred2 = two >= three
    pred3 = two < three
    pred4 = two <= three

    pred5 = false or false
    pred6 = false and false

    assert pred1.create_description() == "This Vertex = Const(2.0) >= three"
    assert pred2.create_description() == "This Vertex = Const(2.0) > three"
    assert pred3.create_description() == "This Vertex = Const(2.0) <= three"
    assert pred4.create_description() == "This Vertex = Const(2.0) < three"
    assert pred5.create_description() == "This Vertex = Const(false) || Const(false)"
    assert pred6.create_description() == "This Vertex = Const(false) && Const(false)"
