## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext().jvm_view()


java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")


def GreaterThan(*args) -> k.GreaterThanVertex:
    return Vertex(k.GreaterThanVertex, args)


def CastDouble(*args) -> k.CastDoubleVertex:
    return Vertex(k.CastDoubleVertex, args)


def DoubleIf(*args) -> k.DoubleIfVertex:
    return Vertex(k.DoubleIfVertex, args)


def Cauchy(*args) -> k.CauchyVertex:
    return Vertex(k.CauchyVertex, args)


def Exponential(*args) -> k.ExponentialVertex:
    return Vertex(k.ExponentialVertex, args)


def Gamma(*args) -> k.GammaVertex:
    return Vertex(k.GammaVertex, args)


def Poisson(*args) -> k.PoissonVertex:
    return Vertex(k.PoissonVertex, args)


def UniformInt(*args) -> k.UniformIntVertex:
    return Vertex(k.UniformIntVertex, args)
