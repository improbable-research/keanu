## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex
from keanu.vartypes import (
    vertex_param_types,
    tensor_arg_types,
    shape_types
)

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
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex")
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
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex")
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
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LogNormalVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.LogisticVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.MultivariateGaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ParetoVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.SmoothUniformVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.StudentTVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.TriangularVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")


def ConstantBool(constant: tensor_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantBoolVertex, constant)


def Equals(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().EqualsVertex, a, b)


def GreaterThanOrEqual(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanOrEqualVertex, a, b)


def GreaterThan(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanVertex, a, b)


def LessThanOrEqual(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanOrEqualVertex, a, b)


def LessThan(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().LessThanVertex, a, b)


def NotEquals(a: vertex_param_types, b: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().NotEqualsVertex, a, b)


def CastDouble(input_vertex: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().CastDoubleVertex, input_vertex)


def ConstantDouble(constant: tensor_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantDoubleVertex, constant)


def DoubleIf(predicate: vertex_param_types, thn: vertex_param_types, els: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().DoubleIfVertex, predicate, thn, els)


def Addition(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Adds one vertex to another
    
    :param left: a vertex to add
    :param right: a vertex to add
    """
    return Vertex(context.jvm_view().AdditionVertex, left, right)


def ArcTan2(x: vertex_param_types, y: vertex_param_types) -> Vertex:
    """
    Calculates the signed angle, in radians, between the positive x-axis and a ray to the point (x, y) from the origin
    
    :param x: x coordinate
    :param y: y coordinate
    """
    return Vertex(context.jvm_view().ArcTan2Vertex, x, y)


def Difference(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Subtracts one vertex from another
    
    :param left: the vertex that will be subtracted from
    :param right: the vertex to subtract
    """
    return Vertex(context.jvm_view().DifferenceVertex, left, right)


def Division(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Divides one vertex by another
    
    :param left: the vertex to be divided
    :param right: the vertex to divide
    """
    return Vertex(context.jvm_view().DivisionVertex, left, right)


def MatrixMultiplication(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Matrix multiplies one vertex by another. C = AB
    
    :param left: vertex A
    :param right: vertex B
    """
    return Vertex(context.jvm_view().MatrixMultiplicationVertex, left, right)


def Max(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Finds the maximum between two vertices
    
    :param left: one of the vertices to find the maximum of
    :param right: one of the vertices to find the maximum of
    """
    return Vertex(context.jvm_view().MaxVertex, left, right)


def Min(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Finds the minimum between two vertices
    
    :param left: one of the vertices to find the minimum of
    :param right: one of the vertices to find the minimum of
    """
    return Vertex(context.jvm_view().MinVertex, left, right)


def Multiplication(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Multiplies one vertex by another
    
    :param left: vertex to be multiplied
    :param right: vertex to be multiplied
    """
    return Vertex(context.jvm_view().MultiplicationVertex, left, right)


def Power(base: vertex_param_types, exponent: vertex_param_types) -> Vertex:
    """
    Raises a vertex to the power of another
    
    :param base: the base vertex
    :param exponent: the exponent vertex
    """
    return Vertex(context.jvm_view().PowerVertex, base, exponent)


def Abs(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the absolute of a vertex
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().AbsVertex, input_vertex)


def ArcCos(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the inverse cosine of a vertex, Arccos(vertex)
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().ArcCosVertex, input_vertex)


def ArcSin(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the inverse sin of a vertex, Arcsin(vertex)
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().ArcSinVertex, input_vertex)


def ArcTan(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the inverse tan of a vertex, Arctan(vertex)
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().ArcTanVertex, input_vertex)


def Ceil(input_vertex: vertex_param_types) -> Vertex:
    """
    Applies the Ceiling operator to a vertex.
    This maps a vertex to the smallest integer greater than or equal to its value
    
    :param input_vertex: the vertex to be ceil'd
    """
    return Vertex(context.jvm_view().CeilVertex, input_vertex)


def Cos(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the cosine of a vertex, Cos(vertex)
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().CosVertex, input_vertex)


def Exp(input_vertex: vertex_param_types) -> Vertex:
    """
    Calculates the exponential of an input vertex
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().ExpVertex, input_vertex)


def Floor(input_vertex: vertex_param_types) -> Vertex:
    """
    Applies the Floor operator to a vertex.
    This maps a vertex to the biggest integer less than or equal to its value
    
    :param input_vertex: the vertex to be floor'd
    """
    return Vertex(context.jvm_view().FloorVertex, input_vertex)


def LogGamma(input_vertex: vertex_param_types) -> Vertex:
    """
    Returns the log of the gamma of the inputVertex
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().LogGammaVertex, input_vertex)


def Log(input_vertex: vertex_param_types) -> Vertex:
    """
    Returns the natural logarithm, base e, of a vertex
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().LogVertex, input_vertex)


def MatrixDeterminant(vertex: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().MatrixDeterminantVertex, vertex)


def MatrixInverse(input_vertex: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().MatrixInverseVertex, input_vertex)


def Reshape(input_vertex: vertex_param_types, proposed_shape: shape_types) -> Vertex:
    return Vertex(context.jvm_view().ReshapeVertex, input_vertex, proposed_shape)


def Round(input_vertex: vertex_param_types) -> Vertex:
    """
    Applies the Rounding operator to a vertex.
    This maps a vertex to the nearest integer value
    
    :param input_vertex: the vertex to be rounded
    """
    return Vertex(context.jvm_view().RoundVertex, input_vertex)


def Sigmoid(input_vertex: vertex_param_types) -> Vertex:
    """
    Applies the sigmoid function to a vertex.
    The sigmoid function is a special case of the Logistic function.
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().SigmoidVertex, input_vertex)


def Sin(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the sine of a vertex. Sin(vertex).
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().SinVertex, input_vertex)


def Sum(input_vertex: vertex_param_types, over_dimensions: shape_types) -> Vertex:
    """
    Performs a sum across specified dimensions. Negative dimension indexing is not supported.
    
    :param input_vertex: the vertex to have its values summed
    :param over_dimensions: dimensions to sum over
    """
    return Vertex(context.jvm_view().SumVertex, input_vertex, over_dimensions)


def Take(input_vertex: vertex_param_types, index: shape_types) -> Vertex:
    """
    A vertex that extracts a scalar at a given index
    
    :param input_vertex: the input vertex to extract from
    :param index: the index to extract at
    """
    return Vertex(context.jvm_view().TakeVertex, input_vertex, index)


def Tan(input_vertex: vertex_param_types) -> Vertex:
    """
    Takes the tangent of a vertex. Tan(vertex).
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().TanVertex, input_vertex)


def Beta(alpha: vertex_param_types, beta: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some tensorShape of alpha and beta to
    a matching tensorShaped Beta.
    
    :param alpha: the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
    :param beta: the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().BetaVertex, alpha, beta)


def Cauchy(location: vertex_param_types, scale: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().CauchyVertex, location, scale)


def ChiSquared(k: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of k to
    a matching shaped ChiSquared.
    
    :param k: the number of degrees of freedom
    """
    return Vertex(context.jvm_view().ChiSquaredVertex, k)


def Dirichlet(concentration: vertex_param_types) -> Vertex:
    """
    Matches a vector of concentration values to a Dirichlet distribution
    
    :param concentration: the concentration values of the dirichlet
    """
    return Vertex(context.jvm_view().DirichletVertex, concentration)


def Exponential(rate: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of rate to matching shaped exponential.
    
    :param rate: the rate of the Exponential with either the same shape as specified for this vertex or scalar
    """
    return Vertex(context.jvm_view().ExponentialVertex, rate)


def Gamma(theta: vertex_param_types, k: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of theta and k to matching shaped gamma.
    
    :param theta: the theta (scale) of the Gamma with either the same shape as specified for this vertex
    :param k: the k (shape) of the Gamma with either the same shape as specified for this vertex
    """
    return Vertex(context.jvm_view().GammaVertex, theta, k)


def Gaussian(mu: vertex_param_types, sigma: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().GaussianVertex, mu, sigma)


def HalfCauchy(scale: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().HalfCauchyVertex, scale)


def HalfGaussian(sigma: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().HalfGaussianVertex, sigma)


def InverseGamma(alpha: vertex_param_types, beta: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of alpha and beta to
    alpha matching shaped Inverse Gamma.
    
    :param alpha: the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    :param beta: the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    """
    return Vertex(context.jvm_view().InverseGammaVertex, alpha, beta)


def Laplace(mu: vertex_param_types, beta: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Laplace.
    
    :param mu: the mu of the Laplace with either the same shape as specified for this vertex or a scalar
    :param beta: the beta of the Laplace with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().LaplaceVertex, mu, beta)


def LogNormal(mu: vertex_param_types, sigma: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().LogNormalVertex, mu, sigma)


def Logistic(mu: vertex_param_types, s: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().LogisticVertex, mu, s)


def MultivariateGaussian(mu: vertex_param_types, covariance: vertex_param_types) -> Vertex:
    """
    Matches a mu and covariance of some shape to a Multivariate Gaussian
    
    :param mu: the mu of the Multivariate Gaussian
    :param covariance: the covariance matrix of the Multivariate Gaussian
    """
    return Vertex(context.jvm_view().MultivariateGaussianVertex, mu, covariance)


def Pareto(location: vertex_param_types, scale: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().ParetoVertex, location, scale)


def SmoothUniform(x_min: vertex_param_types, x_max: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().SmoothUniformVertex, x_min, x_max)


def StudentT(v: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().StudentTVertex, v)


def Triangular(x_min: vertex_param_types, x_max: vertex_param_types, c: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
    
    :param x_min: the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
    :param x_max: the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
    :param c: the c of the Triangular with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().TriangularVertex, x_min, x_max, c)


def Uniform(x_min: vertex_param_types, x_max: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Uniform Vertex
    
    :param x_min: the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
    :param x_max: the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().UniformVertex, x_min, x_max)


def ConstantInteger(constant: tensor_arg_types) -> Vertex:
    return Vertex(context.jvm_view().ConstantIntegerVertex, constant)


def IntegerDivision(left: vertex_param_types, right: vertex_param_types) -> Vertex:
    """
    Divides one vertex by another
    
    :param left: a vertex to be divided
    :param right: a vertex to divide by
    """
    return Vertex(context.jvm_view().IntegerDivisionVertex, left, right)


def Poisson(mu: vertex_param_types) -> Vertex:
    """
    One to one constructor for mapping some shape of mu to
    a matching shaped Poisson.
    
    :param mu: mu with same shape as desired Poisson tensor or scalar
    """
    return Vertex(context.jvm_view().PoissonVertex, mu)


def UniformInt(min: vertex_param_types, max: vertex_param_types) -> Vertex:
    return Vertex(context.jvm_view().UniformIntVertex, min, max)
