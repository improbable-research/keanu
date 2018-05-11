import io.improbable.keanu.network.BayesNet
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient
import java.util.*
import kotlin.math.sqrt

open class Thermometers {
    var u1 : DoubleVertex
    var u2 : DoubleVertex
    var u3 : DoubleVertex
    var err : DoubleVertex
    var temp : DoubleVertex
    var thermo1 : DoubleVertex
    var thermo2 : DoubleVertex
    var thermo1err : DoubleVertex
    var thermo2err : DoubleVertex
    var net : BayesNet
    var random = Random()

    constructor() : this(
            UniformVertex(0.0, 1.0),
            UniformVertex(0.0, 1.0),
            UniformVertex(0.0, 1.0)
    )

    constructor(u1 : DoubleVertex, u2 : DoubleVertex, u3 : DoubleVertex) {
        this.u1 = u1
        this.u2 = u2
        this.u3 = u3
        temp = logistic(u1, 20.0, 5.0)
        thermo1   = logistic(u2, temp, 1.0)
        thermo2   = logistic(u3, temp, 1.0)
        thermo1err = thermo1 - 22.0
        thermo2err = thermo2 - 23.0
        err = thermo1err*thermo1err + thermo2err*thermo2err
//        net = BayesNet(listOf(u1,u2,u3,temp,thermo1,thermo2,err))
        net = BayesNet(err.connectedGraph)
    }


    fun sample() {
        u1.value = u1.sample()
        u2.value = u2.sample()
        u3.value = u3.sample()
        err.lazyEval()
    }


    fun projectToUnitHypercube() {
        if(u1.value > 1.0) u1.value = 2.0-u1.value
        if(u2.value > 1.0) u2.value = 2.0-u2.value
        if(u3.value > 1.0) u3.value = 2.0-u3.value
        if(u1.value < 0.0) u1.value = -u1.value
        if(u2.value < 0.0) u2.value = -u2.value
        if(u3.value < 0.0) u3.value = -u3.value
    }

    fun setState(doubles : DoubleArray) {
        u1.value = doubles[0]
        u2.value = doubles[1]
        u3.value = doubles[2]
        projectToUnitHypercube()
        err.lazyEval()
    }

    fun setState(vec : Vector3D) {
        setState(vec.toArray())
    }

    fun getState() : Vector3D {
        return Vector3D(u1.value, u2.value, u3.value)
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
            doubleArrayOf(
                    err.dualNumber.partialDerivatives.withRespectTo(u1),
                    err.dualNumber.partialDerivatives.withRespectTo(u2),
                    err.dualNumber.partialDerivatives.withRespectTo(u3)
            )
        })
    }

}