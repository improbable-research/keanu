## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.base import KeanuContext, Vertex

k = KeanuContext().jvm_view()


java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex")

def GreaterThan(*args) -> k.GreaterThanVertex:
    return Vertex(k.GreaterThanVertex, args)


java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex")

def CastDouble(*args) -> k.CastDoubleVertex:
    return Vertex(k.CastDoubleVertex, args)


java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex")

def DoubleIf(*args) -> k.DoubleIfVertex:
    return Vertex(k.DoubleIfVertex, args)


java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.CauchyVertex")

def Cauchy(*args) -> k.CauchyVertex:
    return Vertex(k.CauchyVertex, args)


java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex")

def Exponential(*args) -> k.ExponentialVertex:
    return Vertex(k.ExponentialVertex, args)


java_import(k, "io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex")

def Gamma(*args) -> k.GammaVertex:
    return Vertex(k.GammaVertex, args)


java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex")

def Poisson(*args) -> k.PoissonVertex:
    return Vertex(k.PoissonVertex, args)


java_import(k, "io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex")

def UniformInt(*args) -> k.UniformIntVertex:
    return Vertex(k.UniformIntVertex, args)
