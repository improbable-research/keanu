from typing import Any, Dict

import pytest

from keanu.plates import Plates, Plate
from keanu.vertex import Bernoulli, DoubleProxy, Exponential, Poisson, Const, KeanuContext


def test_you_can_iterate_over_the_plates() -> None:
    num_plates = 100

    plates = Plates(count=num_plates, factory=lambda p: None)
    plate_count = 0
    for plate in plates:
        plate_count += 1

    assert plate_count == num_plates


def test_you_can_build_plates_with_fixed_count() -> None:
    num_plates = 100
    vertexLabel = "foo"

    def create_vertex(plate: Plate) -> None:
        v = Bernoulli(0.5)
        v.set_label(vertexLabel)
        plate.add(v)

    plates = Plates(count=num_plates, factory=create_vertex)
    assert plates.size() == num_plates

    for plate in plates:
        assert plate.get(vertexLabel) is not None


def test_you_can_build_plates_from_data() -> None:
    num_plates = 10

    data_generator = ({"x": i, "y": -i} for i in range(num_plates))

    def create_vertices(plate: Plate, point: Dict[str, Any]) -> None:
        plate.add(Const(point["x"], label="x"))
        plate.add(Const(point["y"], label="y"))

    plates = Plates(data_generator=data_generator, factory=create_vertices)
    assert plates.size() == num_plates

    i = 0
    for plate in plates:
        assert plate.get("x").get_value() == i
        assert plate.get("y").get_value() == -i
        i += 1


def test_you_must_pass_count_or_data_generator() -> None:
    with pytest.raises(AssertionError, match="You must specify either a count or a data_generator"):
        Plates(factory=lambda _: None)


def test_you_cannot_pass_both_count_and_data_generator() -> None:
    with pytest.raises(AssertionError, match="You must specify either a count or a data_generator"):
        Plates(factory=lambda _: None, count=1, data_generator=({} for _ in []))


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
    x_previous_label = Plates.proxy_for(x_label)

    num_plates = 10
    initial_x = 1.

    def create_time_step(plate):
        x_previous = DoubleProxy((), x_previous_label)
        x = Exponential(x_previous)
        y = Poisson(x)
        plate.add(x_previous)
        plate.add(x, label=x_label)
        plate.add(y, label=y_label)

    plates = Plates(initial_state={x_label: initial_x}, count=num_plates, factory=create_time_step)
    assert plates.size() == num_plates

    x_from_previous_plate = None
    for plate in plates:
        x_previous_proxy = plate.get(x_previous_label)
        x = plate.get(x_label)
        y = plate.get(y_label)
        if x_from_previous_plate is None:
            assert [p.get_value() for p in x_previous_proxy.get_parents()] == [initial_x]
        else:
            assert [p.get_id() for p in x_previous_proxy.get_parents()] == [x_from_previous_plate.get_id()]
        assert [p.get_id() for p in x.get_parents()] == [x_previous_proxy.get_id()]
        assert [p.get_id() for p in y.get_parents()] == [x.get_id()]
        x_from_previous_plate = x
