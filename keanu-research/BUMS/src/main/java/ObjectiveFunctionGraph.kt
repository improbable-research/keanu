import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient

open abstract class ObjectiveFunctionGraph {
    val state : Array<UniformVertex>

    constructor(dimension : Int) {
        state = Array<UniformVertex>(dimension, {_ -> UniformVertex(0.0,1.0) })

    }

    abstract fun getErrVertex() : DoubleVertex

    fun setState(doubles : DoubleArray) {
        doubles.forEachIndexed { i, v ->
            state[i].value = v
        }
        getErrVertex().lazyEval()
    }

    fun fitness() : ObjectiveFunction {
        return ObjectiveFunction({doubles ->
            setState(doubles)
            getErrVertex().value
        })
    }

    fun gradient() : ObjectiveFunctionGradient {
        return ObjectiveFunctionGradient({doubles ->
            setState(doubles)
            val grad = doubleArrayOf(
                    // TODO: fill this
            )
            grad
        })
    }


}