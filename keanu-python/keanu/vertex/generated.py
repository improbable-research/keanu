## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex
from keanu.vartypes import mypy_vertex_arg_types, mypy_shape_types
import typing
context = KeanuContext()


java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")


def ConstantBool(constant : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantBoolVertex, constant)


def Equals(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().EqualsVertex, a, b)


def GreaterThanOrEqual(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanOrEqualVertex, a, b)


def GreaterThan(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanVertex, a, b)


def LessThanOrEqual(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanOrEqualVertex, a, b)


def LessThan(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanVertex, a, b)


def NotEquals(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().NotEqualsVertex, a, b)


def CastDouble(input_vertex : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CastDoubleVertex, input_vertex)


def ConstantDouble(constant : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantDoubleVertex, constant)


def DoubleIf(shape : mypy_shape_types, predicate : mypy_vertex_arg_types, thn : mypy_vertex_arg_types, els : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DoubleIfVertex, shape, predicate, thn, els)


def Addition(left : mypy_vertex_arg_types, right : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().AdditionVertex, left, right)


def Difference(left : mypy_vertex_arg_types, right : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DifferenceVertex, left, right)


def Division(left : mypy_vertex_arg_types, right : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DivisionVertex, left, right)


def Multiplication(left : mypy_vertex_arg_types, right : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().MultiplicationVertex, left, right)


def Power(base : mypy_vertex_arg_types, exponent : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().PowerVertex, base, exponent)


def Abs(input_vertex : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().AbsVertex, input_vertex)


def Ceil(input_vertex : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CeilVertex, input_vertex)


def Floor(input_vertex : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().FloorVertex, input_vertex)


def Round(input_vertex : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().RoundVertex, input_vertex)


def Cauchy(location : mypy_vertex_arg_types, scale : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CauchyVertex, location, scale)


def Exponential(rate : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ExponentialVertex, rate)


def Gamma(theta : mypy_vertex_arg_types, k : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GammaVertex, theta, k)


def Gaussian(mu : mypy_vertex_arg_types, sigma : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GaussianVertex, mu, sigma)


def Uniform(x_min : mypy_vertex_arg_types, x_max : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().UniformVertex, x_min, x_max)


def ConstantInteger(constant : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantIntegerVertex, constant)


def IntegerDivision(a : mypy_vertex_arg_types, b : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().IntegerDivisionVertex, a, b)


def Poisson(mu : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().PoissonVertex, mu)


def UniformInt(min : mypy_vertex_arg_types, max : mypy_vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().UniformIntVertex, min, max)
