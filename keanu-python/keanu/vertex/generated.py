## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex
from keanu.vartypes import vertex_arg_types, shape_types

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


def ConstantBool(constant : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantBoolVertex, constant)


def Equals(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().EqualsVertex, a, b)


def GreaterThanOrEqual(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanOrEqualVertex, a, b)


def GreaterThan(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanVertex, a, b)


def LessThanOrEqual(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanOrEqualVertex, a, b)


def LessThan(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanVertex, a, b)


def NotEquals(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().NotEqualsVertex, a, b)


def CastDouble(input_vertex : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CastDoubleVertex, input_vertex)


def ConstantDouble(constant : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantDoubleVertex, constant)


def DoubleIf(shape : shape_types, predicate : vertex_arg_types, thn : vertex_arg_types, els : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DoubleIfVertex, shape, predicate, thn, els)


def Addition(left : vertex_arg_types, right : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().AdditionVertex, left, right)


def Difference(left : vertex_arg_types, right : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DifferenceVertex, left, right)


def Division(left : vertex_arg_types, right : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().DivisionVertex, left, right)


def Multiplication(left : vertex_arg_types, right : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().MultiplicationVertex, left, right)


def Power(base : vertex_arg_types, exponent : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().PowerVertex, base, exponent)


def Abs(input_vertex : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().AbsVertex, input_vertex)


def Ceil(input_vertex : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CeilVertex, input_vertex)


def Floor(input_vertex : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().FloorVertex, input_vertex)


def Round(input_vertex : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().RoundVertex, input_vertex)


def Cauchy(location : vertex_arg_types, scale : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().CauchyVertex, location, scale)


def Exponential(rate : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ExponentialVertex, rate)


def Gamma(theta : vertex_arg_types, k : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GammaVertex, theta, k)


def Gaussian(mu : vertex_arg_types, sigma : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().GaussianVertex, mu, sigma)


def Uniform(x_min : vertex_arg_types, x_max : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().UniformVertex, x_min, x_max)


def ConstantInteger(constant : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantIntegerVertex, constant)


def IntegerDivision(a : vertex_arg_types, b : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().IntegerDivisionVertex, a, b)


def Poisson(mu : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().PoissonVertex, mu)


def UniformInt(min : vertex_arg_types, max : vertex_arg_types) -> Vertex:
    return Vertex(context.jvm_view().UniformIntVertex, min, max)
