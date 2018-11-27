from keanu.vertex import Gaussian, LambdaModel, Const


def plus_one(vertices):
    vertices["out"] = vertices["in"] + 1.


class BlackBoxProcess:

    def __init__(self, dir):
        self.out_file = dir.join("out.txt")

    def run(self, vertices):
        out = vertices["in"].get_value() + 1.
        self.out_file.write(out)

    def extract_values(self):
        value = float(self.out_file.read())
        return {"out": Const(value)}


def test_you_can_create_a_lambda_model_vertex():
    v_in = Gaussian(1., 1.)
    model = LambdaModel({"in": v_in}, plus_one)
    run(model, v_in)


def test_you_can_create_a_process_model_vertex(tmpdir):
    dir = tmpdir.mkdir("black_box")
    process = BlackBoxProcess(dir)
    v_in = Gaussian(1., 1.)
    model = LambdaModel({"in": v_in}, process.run, process.extract_values)
    run(model, v_in)


def run(model, v_in):
    v_out = model.get_double_model_output_vertex("out")

    v_in.set_value(1.)
    v_out.eval()
    assert v_out.get_value() == 2.

    v_in.set_value(1.1)
    v_out.eval()
    assert v_out.get_value() == 2.1
