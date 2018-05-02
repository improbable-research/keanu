import io.improbable.keanu.network.BayesNet
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient

class Thermometers {
    var u1 = UniformVertex(0.0,1.0)
    var u2 = UniformVertex(0.0,1.0)
    var u3 = UniformVertex(0.0,1.0)
    var err : DoubleVertex
    var temp : DoubleVertex
    var thermo1 : DoubleVertex
    var thermo2 : DoubleVertex
    var net : BayesNet

    constructor() {
        temp = logistic(u1, 20.0, 5.0)
        thermo1   = logistic(u2, temp, 1.0)
        thermo2   = logistic(u3, temp, 1.0)
        var thermo1err = thermo1 - 22.0
        var thermo2err = thermo2 - 23.0
        err = thermo1err*thermo1err + thermo2err*thermo2err
        net = BayesNet(listOf(u1,u2,u3,temp,thermo1,thermo2,err))
    }

    fun sample() {
        u1.value = u1.sample()
        u2.value = u2.sample()
        u3.value = u3.sample()
        err.lazyEval()
    }

    fun setState(doubles : DoubleArray) {
        u1.value = doubles[0]
        u2.value = doubles[1]
        u3.value = doubles[2]
        err.lazyEval()
    }

    fun fitness() : ObjectiveFunction {
        return ObjectiveFunction({doubles ->
            setState(doubles)
//            println("fitness = ${err.value}")
            err.value
        })
    }

    fun gradient() : ObjectiveFunctionGradient {
        return ObjectiveFunctionGradient({doubles ->
            setState(doubles)
            val grad = doubleArrayOf(
                    err.dualNumber.infinitesimal.infinitesimals[u1.id]!!,
                    err.dualNumber.infinitesimal.infinitesimals[u2.id]!!,
                    err.dualNumber.infinitesimal.infinitesimals[u3.id]!!
            )
//            println("Gradient = ${grad[0]}, ${grad[1]}, ${grad[2]}")
            grad
        })
    }


}