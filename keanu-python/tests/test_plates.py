from keanu.plates import Plates, Plate
from keanu.vertex import Bernoulli


def test_you_can_iterate_over_the_plates() -> None:
    num_plates = 100

    plates = Plates(count=num_plates, factory=lambda p : None)
    plate_count=0
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

def test_you_can_build_plates_from_csv():
    pass

def test_you_can_build_a_time_series():
    pass