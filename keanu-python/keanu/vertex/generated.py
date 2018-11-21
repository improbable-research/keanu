## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex

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
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.BetaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ChiSquaredVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
java_import(context.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex")
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


def ConstantBool(constant) -> Vertex:
    return Vertex(context.jvm_view().ConstantBoolVertex, constant)


def Equals(a, b) -> Vertex:
    return Vertex(context.jvm_view().EqualsVertex, a, b)


def GreaterThanOrEqual(a, b) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanOrEqualVertex, a, b)


def GreaterThan(a, b) -> Vertex:
    return Vertex(context.jvm_view().GreaterThanVertex, a, b)


def LessThanOrEqual(a, b) -> Vertex:
    return Vertex(context.jvm_view().LessThanOrEqualVertex, a, b)


def LessThan(a, b) -> Vertex:
    return Vertex(context.jvm_view().LessThanVertex, a, b)


def NotEquals(a, b) -> Vertex:
    return Vertex(context.jvm_view().NotEqualsVertex, a, b)


def CastDouble(input_vertex) -> Vertex:
    return Vertex(context.jvm_view().CastDoubleVertex, input_vertex)


def ConstantDouble(constant) -> Vertex:
    return Vertex(context.jvm_view().ConstantDoubleVertex, constant)


def DoubleIf(shape, predicate, thn, els) -> Vertex:
    return Vertex(context.jvm_view().DoubleIfVertex, shape, predicate, thn, els)


def Addition(left, right) -> Vertex:
    """
    Adds one vertex to another
    
    :param left: a vertex to add
    :param right: a vertex to add
    """
    return Vertex(context.jvm_view().AdditionVertex, left, right)


def Difference(left, right) -> Vertex:
    """
    Subtracts one vertex from another
    
    :param left: the vertex that will be subtracted from
    :param right: the vertex to subtract
    """
    return Vertex(context.jvm_view().DifferenceVertex, left, right)


def Division(left, right) -> Vertex:
    """
    Divides one vertex by another
    
    :param left: the vertex to be divided
    :param right: the vertex to divide
    """
    return Vertex(context.jvm_view().DivisionVertex, left, right)


def Multiplication(left, right) -> Vertex:
    """
    Multiplies one vertex by another
    
    :param left: vertex to be multiplied
    :param right: vertex to be multiplied
    """
    return Vertex(context.jvm_view().MultiplicationVertex, left, right)


def Power(base, exponent) -> Vertex:
    """
    Raises a vertex to the power of another
    
    :param base: the base vertex
    :param exponent: the exponent vertex
    """
    return Vertex(context.jvm_view().PowerVertex, base, exponent)


def Abs(input_vertex) -> Vertex:
    """
    Takes the absolute of a vertex
    
    :param input_vertex: the vertex
    """
    return Vertex(context.jvm_view().AbsVertex, input_vertex)


def Ceil(input_vertex) -> Vertex:
    """
    Applies the Ceiling operator to a vertex.
    This maps a vertex to the smallest integer greater than or equal to its value
    
    :param input_vertex: the vertex to be ceil'd
    """
    return Vertex(context.jvm_view().CeilVertex, input_vertex)


def Floor(input_vertex) -> Vertex:
    """
    Applies the Floor operator to a vertex.
    This maps a vertex to the biggest integer less than or equal to its value
    
    :param input_vertex: the vertex to be floor'd
    """
    return Vertex(context.jvm_view().FloorVertex, input_vertex)


def Round(input_vertex) -> Vertex:
    """
    Applies the Rounding operator to a vertex.
    This maps a vertex to the nearest integer value
    
    :param input_vertex: the vertex to be rounded
    """
    return Vertex(context.jvm_view().RoundVertex, input_vertex)


def Beta(alpha, beta) -> Vertex:
    """
    One to one constructor for mapping some tensorShape of alpha and beta to
    a matching tensorShaped Beta.
    
    :param alpha: the alpha of the Beta with either the same tensorShape as specified for this vertex or a scalar
    :param beta: the beta of the Beta with either the same tensorShape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().BetaVertex, alpha, beta)


def Cauchy(location, scale) -> Vertex:
    return Vertex(context.jvm_view().CauchyVertex, location, scale)


def ChiSquared(k) -> Vertex:
    """
    One to one constructor for mapping some shape of k to
    a matching shaped ChiSquared.
    
    :param k: the number of degrees of freedom
    """
    return Vertex(context.jvm_view().ChiSquaredVertex, k)


def Dirichlet(concentration) -> Vertex:
    """
    Matches a vector of concentration values to a Dirichlet distribution
    
    :param concentration: the concentration values of the dirichlet
    """
    return Vertex(context.jvm_view().DirichletVertex, concentration)


def Exponential(rate) -> Vertex:
    """
    One to one constructor for mapping some shape of rate to matching shaped exponential.
    
    :param rate: the rate of the Exponential with either the same shape as specified for this vertex or scalar
    """
    return Vertex(context.jvm_view().ExponentialVertex, rate)


def Gamma(theta, k) -> Vertex:
    """
    One to one constructor for mapping some shape of theta and k to matching shaped gamma.
    
    :param theta: the theta (scale) of the Gamma with either the same shape as specified for this vertex
    :param k: the k (shape) of the Gamma with either the same shape as specified for this vertex
    """
    return Vertex(context.jvm_view().GammaVertex, theta, k)


def Gaussian(mu, sigma) -> Vertex:
    return Vertex(context.jvm_view().GaussianVertex, mu, sigma)


def HalfCauchy(scale) -> Vertex:
    return Vertex(context.jvm_view().HalfCauchyVertex, scale)


def InverseGamma(alpha, beta) -> Vertex:
    """
    One to one constructor for mapping some shape of alpha and beta to
    alpha matching shaped Inverse Gamma.
    
    :param alpha: the alpha of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    :param beta: the beta of the Inverse Gamma with either the same shape as specified for this vertex or alpha scalar
    """
    return Vertex(context.jvm_view().InverseGammaVertex, alpha, beta)


def Laplace(mu, beta) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Laplace.
    
    :param mu: the mu of the Laplace with either the same shape as specified for this vertex or a scalar
    :param beta: the beta of the Laplace with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().LaplaceVertex, mu, beta)


def LogNormal(mu, sigma) -> Vertex:
    return Vertex(context.jvm_view().LogNormalVertex, mu, sigma)


def Logistic(mu, s) -> Vertex:
    return Vertex(context.jvm_view().LogisticVertex, mu, s)


def MultivariateGaussian(mu, covariance) -> Vertex:
    """
    Matches a mu and covariance of some shape to a Multivariate Gaussian
    
    :param mu: the mu of the Multivariate Gaussian
    :param covariance: the covariance matrix of the Multivariate Gaussian
    """
    return Vertex(context.jvm_view().MultivariateGaussianVertex, mu, covariance)


def Pareto(location, scale) -> Vertex:
    return Vertex(context.jvm_view().ParetoVertex, location, scale)


def SmoothUniform(x_min, x_max, edge_sharpness) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Smooth Uniform.
    
    :param x_min: the xMin of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
    :param x_max: the xMax of the Smooth Uniform with either the same shape as specified for this vertex or a scalar
    :param edge_sharpness: the edge sharpness of the Smooth Uniform
    """
    return Vertex(context.jvm_view().SmoothUniformVertex, x_min, x_max, edge_sharpness)


def StudentT(v) -> Vertex:
    return Vertex(context.jvm_view().StudentTVertex, v)


def Triangular(x_min, x_max, c) -> Vertex:
    """
    One to one constructor for mapping some shape of xMin, xMax and c to a matching shaped triangular.
    
    :param x_min: the xMin of the Triangular with either the same shape as specified for this vertex or a scalar
    :param x_max: the xMax of the Triangular with either the same shape as specified for this vertex or a scalar
    :param c: the c of the Triangular with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().TriangularVertex, x_min, x_max, c)


def Uniform(x_min, x_max) -> Vertex:
    """
    One to one constructor for mapping some shape of mu and sigma to
    a matching shaped Uniform Vertex
    
    :param x_min: the inclusive lower bound of the Uniform with either the same shape as specified for this vertex or a scalar
    :param x_max: the exclusive upper bound of the Uniform with either the same shape as specified for this vertex or a scalar
    """
    return Vertex(context.jvm_view().UniformVertex, x_min, x_max)


def ConstantInteger(constant) -> Vertex:
    return Vertex(context.jvm_view().ConstantIntegerVertex, constant)


def IntegerDivision(a, b) -> Vertex:
    """
    Divides one vertex by another
    
    :param a: a vertex to be divided
    :param b: a vertex to divide by
    """
    return Vertex(context.jvm_view().IntegerDivisionVertex, a, b)


def Poisson(mu) -> Vertex:
    """
    One to one constructor for mapping some shape of mu to
    a matching shaped Poisson.
    
    :param mu: mu with same shape as desired Poisson tensor or scalar
    """
    return Vertex(context.jvm_view().PoissonVertex, mu)


def UniformInt(min, max) -> Vertex:
    return Vertex(context.jvm_view().UniformIntVertex, min, max)
