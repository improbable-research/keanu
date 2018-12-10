from typing import Dict, Any

from keanu.vertex import Gaussian, LambdaModel, Const, Vertex


def plus_one(vertices):
    vertices["out"] = vertices["in"] + 1.


class BlackBoxProcess:

    def __init__(self, dir: Any) -> None:
        self.out_file = dir.join("out.txt")

    def run(self, vertices: Dict[str, Vertex]) -> None:
        out = vertices["in"].get_value() + 1.
        self.out_file.write(out)

    def extract_values(self) -> Dict[str, Vertex]:
        value = float(self.out_file.read())
        return {"out": Const(value)}


def test_you_can_create_a_lambda_model_vertex() -> None:
    v_in: Vertex = Gaussian(1., 1.)
    model = LambdaModel({"in": v_in}, plus_one)
    evaluate_and_check_for_increment(model, v_in)


def test_you_can_create_a_process_model_vertex(tmpdir) -> None:
    dir = tmpdir.mkdir("black_box")
    process = BlackBoxProcess(dir)
    v_in: Vertex = Gaussian(1., 1.)
    model = LambdaModel({"in": v_in}, process.run, process.extract_values)
    evaluate_and_check_for_increment(model, v_in)


def evaluate_and_check_for_increment(model: LambdaModel, v_in: Vertex) -> None:
    v_out: Vertex = model.get_double_model_output_vertex("out")

    v_in.set_value(1.)
    v_out.eval()
    assert v_out.get_value() == 2.

    v_in.set_value(1.1)
    v_out.eval()
    assert v_out.get_value() == 2.1
