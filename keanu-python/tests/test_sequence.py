from typing import Any, Dict, Optional

import pytest

from keanu.sequence import Sequence, SequenceItem
from keanu.vertex import Bernoulli, DoubleProxy, Exponential, Poisson, Const, ConstantDouble, \
    vertex_constructor_param_types
from keanu.vertex.label import _VertexLabel


def test_you_can_iterate_over_the_sequence() -> None:
    num_items = 100
    sequence = Sequence(count=num_items, factories=lambda p: None)
    item_count = sum(1 for _ in sequence)
    assert item_count == num_items


def test_you_can_build_a_sequence_with_fixed_count() -> None:
    num_items = 100
    vertexLabel = "foo"

    def create_vertex(item: SequenceItem) -> None:
        v = Bernoulli(0.5)
        v.set_label(vertexLabel)
        item.add(v)

    sequence = Sequence(count=num_items, factories=create_vertex)
    assert sequence.size() == num_items

    for item in sequence:
        assert item.get(vertexLabel) is not None


def test_you_can_build_a_sequence_from_data() -> None:
    num_items = 10

    data_generator = ({"x": i, "y": -i} for i in range(num_items))

    def create_vertices(item: SequenceItem, point: Dict[str, Any]) -> None:
        item.add(Const(point["x"], label="x"))
        item.add(Const(point["y"], label="y"))

    sequence = Sequence(data_generator=data_generator, factories=create_vertices)
    assert sequence.size() == num_items

    for index, item in enumerate(sequence):
        assert item.get("x").get_value() == index
        assert item.get("y").get_value() == -index


def test_you_must_pass_count_or_data_generator() -> None:
    with pytest.raises(
            ValueError,
            match="Cannot create a sequence of an unknown size: you must specify either a count of a data_generator"):
        Sequence(factories=lambda _: None)


def test_you_cannot_pass_both_count_and_data_generator() -> None:
    with pytest.raises(ValueError, match="If you pass in a data_generator you cannot also pass in a count"):
        Sequence(factories=lambda _: None, count=1, data_generator=({} for _ in []))


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

    num_items = 10
    initial_x = 1.

    def create_time_step(sequence_item):
        x_previous = sequence_item.add_double_proxy_for(x_label)
        x = Exponential(x_previous)
        y = Poisson(x)
        sequence_item.add(x, label=x_label)
        sequence_item.add(y, label=y_label)

    sequence = Sequence(initial_state={x_label: initial_x}, count=num_items, factories=create_time_step)
    assert sequence.size() == num_items

    x_from_previous_step = None
    x_previous_label = Sequence.proxy_label_for(x_label)

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


def __check_sequence_output_links_to_input(item: SequenceItem, previous_output_label: str,
                                           current_input_label: str) -> None:
    input_children = item.get(previous_output_label).iter_children()
    optional_output_of_previous_timestep = next(input_children, None)

    assert optional_output_of_previous_timestep is not None
    assert next(input_children, None) is None

    assert optional_output_of_previous_timestep.get_label_without_outer_namespace() == current_input_label


def __check_output_equals(sequence: Sequence, label: str, desired_output: float) -> None:
    sequence_unwrapped = sequence.unwrap()
    x1_result = sequence_unwrapped.getLastItem().get(_VertexLabel(label).unwrap())
    assert len(x1_result.getShape()) == 0

    print(x1_result.getValue())
    print(x1_result.getValue().scalar())
    assert x1_result.getValue().scalar() == desired_output


def test_you_can_use_multiple_factories_to_build_sequences() -> None:
    x1_label = "x1"
    x2_label = "x2"
    x3_label = "x3"
    x4_label = "x4"

    two = ConstantDouble(2)
    half = ConstantDouble(0.5)

    def factory1(sequence_item):
        x1_input = sequence_item.add_double_proxy_for(x1_label)
        x2_input = sequence_item.add_double_proxy_for(x2_label)

        x1_output = x1_input * two
        x1_output.set_label(x1_label)
        x3_output = x2_input * two
        x3_output.set_label(x3_label)

        sequence_item.add(x1_output)
        sequence_item.add(x3_output)

    def factory2(sequence_item):
        x3_input = sequence_item.add_double_proxy_for(x3_label)
        x4_input = sequence_item.add_double_proxy_for(x4_label)

        x2_output = x3_input * half
        x2_output.set_label(x2_label)
        x4_output = x4_input * half
        x4_output.set_label(x4_label)

        sequence_item.add(x2_output)
        sequence_item.add(x4_output)

    x1_start = ConstantDouble(4)
    x2_start = ConstantDouble(4)
    x3_start = ConstantDouble(4)
    x4_start = ConstantDouble(4)

    initial_state: Optional[Dict[str, vertex_constructor_param_types]] = {
        x1_label: x1_start,
        x2_label: x2_start,
        x3_label: x3_start,
        x4_label: x4_start
    }
    factories = [factory1, factory2]

    sequence = Sequence(count=5, factories=factories, initial_state=initial_state)

    assert sequence.size() == 5

    x1_input_label = Sequence.proxy_label_for(x1_label)
    x2_input_label = Sequence.proxy_label_for(x2_label)
    x3_input_label = Sequence.proxy_label_for(x3_label)
    x4_input_label = Sequence.proxy_label_for(x4_label)

    for item in sequence:
        __check_sequence_output_links_to_input(item, x1_input_label, x1_label)
        __check_sequence_output_links_to_input(item, x2_input_label, x3_label)
        __check_sequence_output_links_to_input(item, x3_input_label, x2_label)
        __check_sequence_output_links_to_input(item, x4_input_label, x4_label)

    __check_output_equals(sequence, x1_label, 128)
    __check_output_equals(sequence, x2_label, 2)
    __check_output_equals(sequence, x3_label, 8)
    __check_output_equals(sequence, x4_label, 0.125)


def test_last_item_retrieved_correctly() -> None:
    x_label = "x"

    def factory(sequence_item):
        x = sequence_item.add_double_proxy_for(x_label)
        x_out = x * Const(2.0)
        x_out.set_label(x_label)
        sequence_item.add(x_out)

    x_start = ConstantDouble(1.0)
    initial_state: Optional[Dict[str, vertex_constructor_param_types]] = {x_label: x_start}

    sequence = Sequence(count=2, factories=factory, initial_state=initial_state)

    sequence_item_contents = sequence.get_last_item().get_contents()
    x_output = sequence_item_contents.get(x_label)
    x_proxy = sequence_item_contents.get(Sequence.proxy_label_for(x_label))

    assert x_output is not None
    assert x_proxy is not None
    assert x_output.get_value() == 4
    assert x_proxy.get_value() == 2
