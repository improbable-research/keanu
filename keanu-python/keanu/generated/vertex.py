## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext().jvm_view()


java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex")
java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")
java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")


def ConstantBool(*args) -> k.ConstantBoolVertex:
    return Vertex(k.ConstantBoolVertex, args)


def GreaterThan(*args) -> k.GreaterThanVertex:
    return Vertex(k.GreaterThanVertex, args)


def LessThan(*args) -> k.LessThanVertex:
    return Vertex(k.LessThanVertex, args)


def CastDouble(*args) -> k.CastDoubleVertex:
    return Vertex(k.CastDoubleVertex, args)


def ConstantDouble(*args) -> k.ConstantDoubleVertex:
    return Vertex(k.ConstantDoubleVertex, args)


def DoubleIf(*args) -> k.DoubleIfVertex:
    return Vertex(k.DoubleIfVertex, args)


def Addition(*args) -> k.AdditionVertex:
    return Vertex(k.AdditionVertex, args)


def Difference(*args) -> k.DifferenceVertex:
    return Vertex(k.DifferenceVertex, args)


def Division(*args) -> k.DivisionVertex:
    return Vertex(k.DivisionVertex, args)


def Multiplication(*args) -> k.MultiplicationVertex:
    return Vertex(k.MultiplicationVertex, args)


def Cauchy(*args) -> k.CauchyVertex:
    return Vertex(k.CauchyVertex, args)


def Exponential(*args) -> k.ExponentialVertex:
    return Vertex(k.ExponentialVertex, args)


def Gamma(*args) -> k.GammaVertex:
    return Vertex(k.GammaVertex, args)


def Gaussian(*args) -> k.GaussianVertex:
    return Vertex(k.GaussianVertex, args)


def Uniform(*args) -> k.UniformVertex:
    return Vertex(k.UniformVertex, args)


def ConstantInteger(*args) -> k.ConstantIntegerVertex:
    return Vertex(k.ConstantIntegerVertex, args)


def Poisson(*args) -> k.PoissonVertex:
    return Vertex(k.PoissonVertex, args)


def UniformInt(*args) -> k.UniformIntVertex:
    return Vertex(k.UniformIntVertex, args)
