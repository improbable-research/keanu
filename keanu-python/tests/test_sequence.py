from typing import Any, Dict

import pytest

from keanu.sequence import Sequence, SequenceItem
from keanu.vertex import Bernoulli, DoubleProxy, Exponential, Poisson, Const, KeanuContext


def test_you_can_iterate_over_the_sequence() -> None:
    num_items = 100
    sequence = Sequence(count=num_items, factory=lambda p: None)
    item_count = sum(1 for _ in sequence)
    assert item_count == num_items


def test_you_can_build_a_sequence_with_fixed_count() -> None:
    num_items = 100
    vertexLabel = "foo"

    def create_vertex(item: SequenceItem) -> None:
        v = Bernoulli(0.5)
        v.set_label(vertexLabel)
        item.add(v)

    sequence = Sequence(count=num_items, factory=create_vertex)
    assert sequence.size() == num_items

    for item in sequence:
        assert item.get(vertexLabel) is not None


def test_you_can_build_a_sequence_from_data() -> None:
    num_items = 10

    data_generator = ({"x": i, "y": -i} for i in range(num_items))

    def create_vertices(item: SequenceItem, point: Dict[str, Any]) -> None:
        item.add(Const(point["x"], label="x"))
        item.add(Const(point["y"], label="y"))

    sequence = Sequence(data_generator=data_generator, factory=create_vertices)
    assert sequence.size() == num_items

    for index, item in enumerate(sequence):
        assert item.get("x").get_value() == index
        assert item.get("y").get_value() == -index


def test_you_must_pass_count_or_data_generator() -> None:
    with pytest.raises(
            ValueError,
            match="Cannot create a sequence of an unknown size: you must specify either a count of a data_generator"):
        Sequence(factory=lambda _: None)


def test_you_cannot_pass_both_count_and_data_generator() -> None:
    with pytest.raises(ValueError, match="If you pass in a data_generator you cannot also pass in a count"):
        Sequence(factory=lambda _: None, count=1, data_generator=({} for _ in []))


def test_you_can_build_a_time_series() -> None:
    """
    This is a Hidden Markov Model -
    see for example http://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf

    ...  -->  X[t-1]  -->  X[t]  --> ...
                |           |
              Y[t-1]       Y[t]
    """
    x_label = "x"
    y_label = "y"
    x_previous_label = Sequence.proxy_for(x_label)

    num_items = 10
    initial_x = 1.

    def create_time_step(sequence_item):
        x_previous = DoubleProxy((), x_previous_label)
        x = Exponential(x_previous)
        y = Poisson(x)
        sequence_item.add(x_previous)
        sequence_item.add(x, label=x_label)
        sequence_item.add(y, label=y_label)

    sequence = Sequence(initial_state={x_label: initial_x}, count=num_items, factory=create_time_step)
    assert sequence.size() == num_items

    x_from_previous_step = None
    for item in sequence:
        x_previous_proxy = item.get(x_previous_label)
        x = item.get(x_label)
        y = item.get(y_label)
        if x_from_previous_step is None:
            assert [p.get_value() for p in x_previous_proxy.iter_parents()] == [initial_x]
        else:
            assert [p.get_id() for p in x_previous_proxy.iter_parents()] == [x_from_previous_step.get_id()]
        assert [p.get_id() for p in x.iter_parents()] == [x_previous_proxy.get_id()]
        assert [p.get_id() for p in y.iter_parents()] == [x.get_id()]
        x_from_previous_step = x
