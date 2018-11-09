from keanu.vertex import Gaussian, LambdaModel


def plus_one(vertices): 
    vertices["out"] = vertices["in"] + 1.

def test_you_can_create_a_lambda_model_vertex():
    v_in = Gaussian(1., 1.)

    model = LambdaModel(
        { "in": v_in }, 
        plus_one
    )

    v_out = model.get_double_model_output_vertex("out")

    v_in.set_value(1.)
    v_out.eval()
    assert v_out.get_value() == 2.

    v_in.set_value(1.1)
    v_out.eval()
    assert v_out.get_value() == 2.1
