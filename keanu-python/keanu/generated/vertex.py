## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext()


java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")


def ConstantBool(*args) -> k.jvm_view().ConstantBoolVertex:
    return Vertex(k.jvm_view().ConstantBoolVertex, args)


def Equals(*args) -> k.jvm_view().EqualsVertex:
    return Vertex(k.jvm_view().EqualsVertex, args)


def GreaterThanOrEqual(*args) -> k.jvm_view().GreaterThanOrEqualVertex:
    return Vertex(k.jvm_view().GreaterThanOrEqualVertex, args)


def GreaterThan(*args) -> k.jvm_view().GreaterThanVertex:
    return Vertex(k.jvm_view().GreaterThanVertex, args)


def LessThanOrEqual(*args) -> k.jvm_view().LessThanOrEqualVertex:
    return Vertex(k.jvm_view().LessThanOrEqualVertex, args)


def LessThan(*args) -> k.jvm_view().LessThanVertex:
    return Vertex(k.jvm_view().LessThanVertex, args)


def NotEquals(*args) -> k.jvm_view().NotEqualsVertex:
    return Vertex(k.jvm_view().NotEqualsVertex, args)


def CastDouble(*args) -> k.jvm_view().CastDoubleVertex:
    return Vertex(k.jvm_view().CastDoubleVertex, args)


def ConstantDouble(*args) -> k.jvm_view().ConstantDoubleVertex:
    return Vertex(k.jvm_view().ConstantDoubleVertex, args)


def DoubleIf(*args) -> k.jvm_view().DoubleIfVertex:
    return Vertex(k.jvm_view().DoubleIfVertex, args)


def Addition(*args) -> k.jvm_view().AdditionVertex:
    return Vertex(k.jvm_view().AdditionVertex, args)


def Difference(*args) -> k.jvm_view().DifferenceVertex:
    return Vertex(k.jvm_view().DifferenceVertex, args)


def Division(*args) -> k.jvm_view().DivisionVertex:
    return Vertex(k.jvm_view().DivisionVertex, args)


def Multiplication(*args) -> k.jvm_view().MultiplicationVertex:
    return Vertex(k.jvm_view().MultiplicationVertex, args)


def Power(*args) -> k.jvm_view().PowerVertex:
    return Vertex(k.jvm_view().PowerVertex, args)


def Abs(*args) -> k.jvm_view().AbsVertex:
    return Vertex(k.jvm_view().AbsVertex, args)


def Ceil(*args) -> k.jvm_view().CeilVertex:
    return Vertex(k.jvm_view().CeilVertex, args)


def Floor(*args) -> k.jvm_view().FloorVertex:
    return Vertex(k.jvm_view().FloorVertex, args)


def Round(*args) -> k.jvm_view().RoundVertex:
    return Vertex(k.jvm_view().RoundVertex, args)


def Cauchy(*args) -> k.jvm_view().CauchyVertex:
    return Vertex(k.jvm_view().CauchyVertex, args)


def Exponential(*args) -> k.jvm_view().ExponentialVertex:
    return Vertex(k.jvm_view().ExponentialVertex, args)


def Gamma(*args) -> k.jvm_view().GammaVertex:
    return Vertex(k.jvm_view().GammaVertex, args)


def Gaussian(*args) -> k.jvm_view().GaussianVertex:
    return Vertex(k.jvm_view().GaussianVertex, args)


def Uniform(*args) -> k.jvm_view().UniformVertex:
    return Vertex(k.jvm_view().UniformVertex, args)


def ConstantInteger(*args) -> k.jvm_view().ConstantIntegerVertex:
    return Vertex(k.jvm_view().ConstantIntegerVertex, args)


def Poisson(*args) -> k.jvm_view().PoissonVertex:
    return Vertex(k.jvm_view().PoissonVertex, args)


def UniformInt(*args) -> k.jvm_view().UniformIntVertex:
    return Vertex(k.jvm_view().UniformIntVertex, args)
