## This is a generated file. DO NOT EDIT.

from typing import Collection, Optional
from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex, Double, Integer, Boolean, vertex_constructor_param_types
from keanu.vartypes import (
    tensor_arg_types,
    shape_types
)
from .vertex_casting import (
    do_vertex_cast,
    do_inferred_vertex_cast,
    cast_to_double_tensor,
    cast_to_integer_tensor,
    cast_to_boolean_tensor,
    cast_to_double,
    cast_to_integer,
    cast_to_boolean,
    cast_to_long_array,
    cast_to_int_array,
    cast_to_vertex_array,
)

context = KeanuContext()


def cast_to_double_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantDouble, input)


def cast_to_integer_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantInteger, input)


def cast_to_boolean_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantBoolean, input)


def cast_to_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_inferred_vertex_cast({bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble}, input)


java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanIfVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanProxyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.CastToBooleanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.NumericalEqualsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.AndBinaryVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.OrBinaryVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple.BooleanConcatenationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanReshapeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanSliceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.BooleanTakeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastToDoubleVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanMaskVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanOrEqualToMaskVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanMaskVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanOrEqualToMaskVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary.DoubleSetWithMaskVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogGammaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixDeterminantVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixInverseVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.PermuteVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SliceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.BetaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ChiSquaredVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.InverseGammaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LogNormalVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LogisticVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.MultivariateGaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ParetoVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.SmoothUniformVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.StudentTVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.TriangularVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.generic.nonprobabilistic.PrintVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.CastToIntegerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerIfVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerReshapeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSliceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerTakeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.GeometricVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.MultinomialVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.utility.AssertVertex")


def BooleanIf(predicate: vertex_constructor_param_types, thn: vertex_constructor_param_types, els: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().BooleanIfVertex, label, cast_to_boolean_vertex(predicate), cast_to_boolean_vertex(thn), cast_to_boolean_vertex(els))


def BooleanProxy(shape: Collection[int], label: str) -> Vertex:
    return Boolean(context.jvm_view().BooleanProxyVertex, label, cast_to_long_array(shape), label)


def CastToBoolean(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().CastToBooleanVertex, label, cast_to_vertex(input_vertex))


def ConstantBoolean(constant: tensor_arg_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().ConstantBooleanVertex, label, cast_to_boolean_tensor(constant))


def NumericalEquals(a: vertex_constructor_param_types, b: vertex_constructor_param_types, epsilon: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().NumericalEqualsVertex, label, cast_to_vertex(a), cast_to_vertex(b), cast_to_vertex(epsilon))


def AndBinary(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().AndBinaryVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def OrBinary(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().OrBinaryVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def Equals(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().EqualsVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def GreaterThanOrEqual(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().GreaterThanOrEqualVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def GreaterThan(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().GreaterThanVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def LessThanOrEqual(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().LessThanOrEqualVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def LessThan(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().LessThanVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def NotEquals(a: vertex_constructor_param_types, b: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().NotEqualsVertex, label, cast_to_vertex(a), cast_to_vertex(b))


def BooleanConcatenation(dimension: int, input: Collection[Vertex], label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().BooleanConcatenationVertex, label, cast_to_integer(dimension), cast_to_vertex_array(input))


def BooleanReshape(input_vertex: vertex_constructor_param_types, proposed_shape: Collection[int], label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().BooleanReshapeVertex, label, cast_to_vertex(input_vertex), cast_to_long_array(proposed_shape))


def BooleanSlice(input_vertex: vertex_constructor_param_types, dimension: int, index: int, label: Optional[str]=None) -> Vertex:
    """
    Takes the slice along a given dimension and index of a vertex
    
    :param input_vertex: the input vertex
    :param dimension: the dimension to extract along
    :param index: the index of extraction
    """
    return Boolean(context.jvm_view().BooleanSliceVertex, label, cast_to_vertex(input_vertex), cast_to_integer(dimension), cast_to_integer(index))


def BooleanTake(input_vertex: vertex_constructor_param_types, index: Collection[int], label: Optional[str]=None) -> Vertex:
    """
    A vertex that extracts a scalar at a given index
    
    :param input_vertex: the input vertex to extract from
    :param index: the index to extract at
    """
    return Boolean(context.jvm_view().BooleanTakeVertex, label, cast_to_vertex(input_vertex), cast_to_long_array(index))


def Not(a: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Boolean(context.jvm_view().NotVertex, label, cast_to_vertex(a))


def Bernoulli(prob_true: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of probTrue to
    a matching shaped Bernoulli.
    
    :param prob_true: probTrue with same shape as desired Bernoulli tensor or scalar
    """
    return Boolean(context.jvm_view().BernoulliVertex, label, cast_to_double_vertex(prob_true))


def CastToDouble(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().CastToDoubleVertex, label, cast_to_vertex(input_vertex))


def ConstantDouble(constant: tensor_arg_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().ConstantDoubleVertex, label, cast_to_double_tensor(constant))


def DoubleIf(predicate: vertex_constructor_param_types, thn: vertex_constructor_param_types, els: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleIfVertex, label, cast_to_boolean_vertex(predicate), cast_to_double_vertex(thn), cast_to_double_vertex(els))


def DoubleProxy(shape: Collection[int], label: str) -> Vertex:
    return Double(context.jvm_view().DoubleProxyVertex, label, cast_to_long_array(shape), label)


def Addition(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Adds one vertex to another
    
    :param left: a vertex to add
    :param right: a vertex to add
    """
    return Double(context.jvm_view().AdditionVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def ArcTan2(x: vertex_constructor_param_types, y: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
    
    :param x: x coordinate
    :param y: y coordinate
    """
    return Double(context.jvm_view().ArcTan2Vertex, label, cast_to_double_vertex(x), cast_to_double_vertex(y))


def Difference(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Subtracts one vertex from another
    
    :param left: the vertex that will be subtracted from
    :param right: the vertex to subtract
    """
    return Double(context.jvm_view().DifferenceVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def Division(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Divides one vertex by another
    
    :param left: the vertex to be divided
    :param right: the vertex to divide
    """
    return Double(context.jvm_view().DivisionVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def DoubleGreaterThanMask(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleGreaterThanMaskVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def DoubleGreaterThanOrEqualToMask(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleGreaterThanOrEqualToMaskVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def DoubleLessThanMask(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleLessThanMaskVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def DoubleLessThanOrEqualToMask(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleLessThanOrEqualToMaskVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def MatrixMultiplication(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Matrix multiplies one vertex by another. C = AB
    
    :param left: vertex A
    :param right: vertex B
    """
    return Double(context.jvm_view().MatrixMultiplicationVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def Max(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Finds the maximum between two vertices
    
    :param left: one of the vertices to find the maximum of
    :param right: one of the vertices to find the maximum of
    """
    return Double(context.jvm_view().MaxVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def Min(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Finds the minimum between two vertices
    
    :param left: one of the vertices to find the minimum of
    :param right: one of the vertices to find the minimum of
    """
    return Double(context.jvm_view().MinVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def Multiplication(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Multiplies one vertex by another
    
    :param left: vertex to be multiplied
    :param right: vertex to be multiplied
    """
    return Double(context.jvm_view().MultiplicationVertex, label, cast_to_double_vertex(left), cast_to_double_vertex(right))


def Power(base: vertex_constructor_param_types, exponent: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Raises a vertex to the power of another
    
    :param base: the base vertex
    :param exponent: the exponent vertex
    """
    return Double(context.jvm_view().PowerVertex, label, cast_to_double_vertex(base), cast_to_double_vertex(exponent))


def Concatenation(dimension: int, operands: Collection[Vertex], label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().ConcatenationVertex, label, cast_to_integer(dimension), cast_to_vertex_array(operands))


def DoubleSetWithMask(operand: vertex_constructor_param_types, mask: vertex_constructor_param_types, set_value: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().DoubleSetWithMaskVertex, label, cast_to_double_vertex(operand), cast_to_double_vertex(mask), cast_to_double_vertex(set_value))


def Abs(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the absolute of a vertex
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().AbsVertex, label, cast_to_double_vertex(input_vertex))


def ArcCos(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the inverse cosine of a vertex, Arccos(vertex)
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().ArcCosVertex, label, cast_to_double_vertex(input_vertex))


def ArcSin(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the inverse sin of a vertex, Arcsin(vertex)
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().ArcSinVertex, label, cast_to_double_vertex(input_vertex))


def ArcTan(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the inverse tan of a vertex, Arctan(vertex)
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().ArcTanVertex, label, cast_to_double_vertex(input_vertex))


def Ceil(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Applies the Ceiling operator to a vertex.
    This maps a vertex to the smallest integer greater than or equal to its value
    
    :param input_vertex: the vertex to be ceil'd
    """
    return Double(context.jvm_view().CeilVertex, label, cast_to_double_vertex(input_vertex))


def Cos(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the cosine of a vertex, Cos(vertex)
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().CosVertex, label, cast_to_double_vertex(input_vertex))


def Exp(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Calculates the exponential of an input vertex
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().ExpVertex, label, cast_to_double_vertex(input_vertex))


def Floor(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Applies the Floor operator to a vertex.
    This maps a vertex to the biggest integer less than or equal to its value
    
    :param input_vertex: the vertex to be floor'd
    """
    return Double(context.jvm_view().FloorVertex, label, cast_to_double_vertex(input_vertex))


def LogGamma(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Returns the log of the gamma of the inputVertex
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().LogGammaVertex, label, cast_to_double_vertex(input_vertex))


def Log(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Returns the natural logarithm, base e, of a vertex
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().LogVertex, label, cast_to_double_vertex(input_vertex))


def MatrixDeterminant(vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().MatrixDeterminantVertex, label, cast_to_double_vertex(vertex))


def MatrixInverse(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().MatrixInverseVertex, label, cast_to_double_vertex(input_vertex))


def Permute(input_vertex: vertex_constructor_param_types, rearrange: Collection[int], label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().PermuteVertex, label, cast_to_double_vertex(input_vertex), cast_to_int_array(rearrange))


def Reshape(input_vertex: vertex_constructor_param_types, proposed_shape: Collection[int], label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().ReshapeVertex, label, cast_to_double_vertex(input_vertex), cast_to_long_array(proposed_shape))


def Round(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Applies the Rounding operator to a vertex.
    This maps a vertex to the nearest integer value
    
    :param input_vertex: the vertex to be rounded
    """
    return Double(context.jvm_view().RoundVertex, label, cast_to_double_vertex(input_vertex))


def Sigmoid(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Applies the sigmoid function to a vertex.
    The sigmoid function is a special case of the Logistic function.
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().SigmoidVertex, label, cast_to_double_vertex(input_vertex))


def Sin(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the sine of a vertex. Sin(vertex).
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().SinVertex, label, cast_to_double_vertex(input_vertex))


def Slice(input_vertex: vertex_constructor_param_types, dimension: int, index: int, label: Optional[str]=None) -> Vertex:
    """
    Takes the slice along a given dimension and index of a vertex
    
    :param input_vertex: the input vertex
    :param dimension: the dimension to extract along
    :param index: the index of extraction
    """
    return Double(context.jvm_view().SliceVertex, label, cast_to_double_vertex(input_vertex), cast_to_integer(dimension), cast_to_integer(index))


def Sum(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Performs a sum across all dimensions
    
    :param input_vertex: the vertex to have its values summed
    """
    return Double(context.jvm_view().SumVertex, label, cast_to_double_vertex(input_vertex))


def Take(input_vertex: vertex_constructor_param_types, index: Collection[int], label: Optional[str]=None) -> Vertex:
    """
    A vertex that extracts a scalar at a given index
    
    :param input_vertex: the input vertex to extract from
    :param index: the index to extract at
    """
    return Double(context.jvm_view().TakeVertex, label, cast_to_double_vertex(input_vertex), cast_to_long_array(index))


def Tan(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the tangent of a vertex. Tan(vertex).
    
    :param input_vertex: the vertex
    """
    return Double(context.jvm_view().TanVertex, label, cast_to_double_vertex(input_vertex))


def Beta(alpha: vertex_constructor_param_types, beta: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some tensorShape of alpha and beta to
    a matching tensorShaped Beta.
    
    :param alpha: the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
    :param beta: the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
    """
    return Double(context.jvm_view().BetaVertex, label, cast_to_double_vertex(alpha), cast_to_double_vertex(beta))


def Cauchy(location: vertex_constructor_param_types, scale: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().CauchyVertex, label, cast_to_double_vertex(location), cast_to_double_vertex(scale))


def ChiSquared(k: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of k to
    a matching shaped ChiSquared.
    
    :param k: the number of degrees of freedom
    """
    return Double(context.jvm_view().ChiSquaredVertex, label, cast_to_integer_vertex(k))


def Dirichlet(concentration: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Matches a vector of concentration values to a Dirichlet distribution
    
    :param concentration: the concentration values of the dirichlet
    """
    return Double(context.jvm_view().DirichletVertex, label, cast_to_double_vertex(concentration))


def Exponential(rate: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of rate to matching shaped exponential.
    
    :param rate: the rate of the Exponential with either the same shape as specified for this vertex or scalar
    """
    return Double(context.jvm_view().ExponentialVertex, label, cast_to_double_vertex(rate))


def Gamma(theta: vertex_constructor_param_types, k: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of theta and k to matching shaped gamma.
    
    :param theta: the theta (scale) of the Gamma with either the same shape as specified for this vertex
    :param k: the k (shape) of the Gamma with either the same shape as specified for this vertex
    """
    return Double(context.jvm_view().GammaVertex, label, cast_to_double_vertex(theta), cast_to_double_vertex(k))


def Gaussian(mu: vertex_constructor_param_types, sigma: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().GaussianVertex, label, cast_to_double_vertex(mu), cast_to_double_vertex(sigma))


def HalfCauchy(scale: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().HalfCauchyVertex, label, cast_to_double_vertex(scale))


def HalfGaussian(sigma: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().HalfGaussianVertex, label, cast_to_double_vertex(sigma))


def InverseGamma(alpha: vertex_constructor_param_types, beta: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of alpha and beta to
    alpha matching shaped Inverse Gamma.
    
    :param alpha: the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    :param beta: the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    """
    return Double(context.jvm_view().InverseGammaVertex, label, cast_to_double_vertex(alpha), cast_to_double_vertex(beta))


def KDE(samples: tensor_arg_types, bandwidth: float, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().KDEVertex, label, cast_to_double_tensor(samples), cast_to_double(bandwidth))


def Laplace(mu: vertex_constructor_param_types, beta: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Laplace.
    
    :param mu: the mu of the Laplace with either the same shape as specified for this vertex or a scalar
    :param beta: the beta of the Laplace with either the same shape as specified for this vertex or a scalar
    """
    return Double(context.jvm_view().LaplaceVertex, label, cast_to_double_vertex(mu), cast_to_double_vertex(beta))


def LogNormal(mu: vertex_constructor_param_types, sigma: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().LogNormalVertex, label, cast_to_double_vertex(mu), cast_to_double_vertex(sigma))


def Logistic(mu: vertex_constructor_param_types, s: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().LogisticVertex, label, cast_to_double_vertex(mu), cast_to_double_vertex(s))


def MultivariateGaussian(mu: vertex_constructor_param_types, covariance: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Matches a mu and covariance of some shape to a Multivariate Gaussian
    
    :param mu: the mu of the Multivariate Gaussian
    :param covariance: the covariance matrix of the Multivariate Gaussian
    """
    return Double(context.jvm_view().MultivariateGaussianVertex, label, cast_to_double_vertex(mu), cast_to_double_vertex(covariance))


def Pareto(location: vertex_constructor_param_types, scale: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().ParetoVertex, label, cast_to_double_vertex(location), cast_to_double_vertex(scale))


def SmoothUniform(x_min: vertex_constructor_param_types, x_max: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().SmoothUniformVertex, label, cast_to_double_vertex(x_min), cast_to_double_vertex(x_max))


def StudentT(v: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Double(context.jvm_view().StudentTVertex, label, cast_to_integer_vertex(v))


def Triangular(x_min: vertex_constructor_param_types, x_max: vertex_constructor_param_types, c: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
    
    :param x_min: the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
    :param x_max: the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
    :param c: the c of the Triangular with either the same shape as specified for this vertex or a scalar
    """
    return Double(context.jvm_view().TriangularVertex, label, cast_to_double_vertex(x_min), cast_to_double_vertex(x_max), cast_to_double_vertex(c))


def Uniform(x_min: vertex_constructor_param_types, x_max: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Uniform Vertex
    
    :param x_min: the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
    :param x_max: the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
    """
    return Double(context.jvm_view().UniformVertex, label, cast_to_double_vertex(x_min), cast_to_double_vertex(x_max))


def Print(parent: vertex_constructor_param_types, message: str, print_data: bool, label: Optional[str]=None) -> Vertex:
    return Vertex(context.jvm_view().PrintVertex, label, cast_to_vertex(parent), message, cast_to_boolean(print_data))


def CastToInteger(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().CastToIntegerVertex, label, cast_to_vertex(input_vertex))


def ConstantInteger(constant: tensor_arg_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().ConstantIntegerVertex, label, cast_to_integer_tensor(constant))


def IntegerIf(predicate: vertex_constructor_param_types, thn: vertex_constructor_param_types, els: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().IntegerIfVertex, label, cast_to_boolean_vertex(predicate), cast_to_integer_vertex(thn), cast_to_integer_vertex(els))


def IntegerProxy(tensor_shape: Collection[int], label: str) -> Vertex:
    return Integer(context.jvm_view().IntegerProxyVertex, label, cast_to_long_array(tensor_shape), label)


def IntegerAddition(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Adds one vertex to another
    
    :param left: a vertex to add
    :param right: a vertex to add
    """
    return Integer(context.jvm_view().IntegerAdditionVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerDifference(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Subtracts one vertex from another
    
    :param left: the vertex to be subtracted from
    :param right: the vertex to subtract
    """
    return Integer(context.jvm_view().IntegerDifferenceVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerDivision(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Divides one vertex by another
    
    :param left: a vertex to be divided
    :param right: a vertex to divide by
    """
    return Integer(context.jvm_view().IntegerDivisionVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerMax(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Finds the maximum between two vertices
    
    :param left: one of the vertices to find the maximum of
    :param right: one of the vertices to find the maximum of
    """
    return Integer(context.jvm_view().IntegerMaxVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerMin(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Finds the minimum between two vertices
    
    :param left: one of the vertices to find the minimum of
    :param right: one of the vertices to find the minimum of
    """
    return Integer(context.jvm_view().IntegerMinVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerMultiplication(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Multiplies one vertex by another
    
    :param left: a vertex to be multiplied
    :param right: a vertex to be multiplied
    """
    return Integer(context.jvm_view().IntegerMultiplicationVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerPower(left: vertex_constructor_param_types, right: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Raises one vertex to the power of another
    
    :param left: the base vertex
    :param right: the exponent vertex
    """
    return Integer(context.jvm_view().IntegerPowerVertex, label, cast_to_integer_vertex(left), cast_to_integer_vertex(right))


def IntegerConcatenation(dimension: int, input: Collection[Vertex], label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().IntegerConcatenationVertex, label, cast_to_integer(dimension), cast_to_vertex_array(input))


def IntegerAbs(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Takes the absolute value of a vertex
    
    :param input_vertex: the vertex
    """
    return Integer(context.jvm_view().IntegerAbsVertex, label, cast_to_integer_vertex(input_vertex))


def IntegerReshape(input_vertex: vertex_constructor_param_types, proposed_shape: Collection[int], label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().IntegerReshapeVertex, label, cast_to_integer_vertex(input_vertex), cast_to_long_array(proposed_shape))


def IntegerSlice(input_vertex: vertex_constructor_param_types, dimension: int, index: int, label: Optional[str]=None) -> Vertex:
    """
    Takes the slice along a given dimension and index of a vertex
    
    :param input_vertex: the input vertex
    :param dimension: the dimension to extract along
    :param index: the index of extraction
    """
    return Integer(context.jvm_view().IntegerSliceVertex, label, cast_to_integer_vertex(input_vertex), cast_to_integer(dimension), cast_to_integer(index))


def IntegerSum(input_vertex: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    Performs a sum across all dimensions
    
    :param input_vertex: the vertex to have its values summed
    """
    return Integer(context.jvm_view().IntegerSumVertex, label, cast_to_integer_vertex(input_vertex))


def IntegerTake(input_vertex: vertex_constructor_param_types, index: Collection[int], label: Optional[str]=None) -> Vertex:
    """
    A vertex that extracts a scalar at a given index
    
    :param input_vertex: the input vertex to extract from
    :param index: the index to extract at
    """
    return Integer(context.jvm_view().IntegerTakeVertex, label, cast_to_integer_vertex(input_vertex), cast_to_long_array(index))


def Binomial(p: vertex_constructor_param_types, n: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().BinomialVertex, label, cast_to_double_vertex(p), cast_to_integer_vertex(n))


def Geometric(p: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().GeometricVertex, label, cast_to_double_vertex(p))


def Multinomial(n: vertex_constructor_param_types, p: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().MultinomialVertex, label, cast_to_integer_vertex(n), cast_to_double_vertex(p))


def Poisson(mu: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    """
    One to one constructor for mapping some shape of mu to
    a matching shaped Poisson.
    
    :param mu: mu with same shape as desired Poisson tensor or scalar
    """
    return Integer(context.jvm_view().PoissonVertex, label, cast_to_double_vertex(mu))


def UniformInt(min: vertex_constructor_param_types, max: vertex_constructor_param_types, label: Optional[str]=None) -> Vertex:
    return Integer(context.jvm_view().UniformIntVertex, label, cast_to_integer_vertex(min), cast_to_integer_vertex(max))


def Assert(predicate: vertex_constructor_param_types, error_message: str, label: Optional[str]=None) -> Vertex:
    """
    A vertex that asserts a {@link BooleanVertex} is all true on calculation.
    
    :param predicate: the predicate to evaluate
    :param error_message: a message to include in the {@link AssertionError}
    """
    return Boolean(context.jvm_view().AssertVertex, label, cast_to_vertex(predicate), error_message)
