import io.improbable.keanu.vertices.dbl.DoubleVertex
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient

class UnitHypercubeOptimiser(state: Array<DoubleVertex>, objective: DoubleVertex) : GraphOptimiser(state, objective) {
    override fun fitness() : ObjectiveFunction {
        return ObjectiveFunction({doubles ->
            setState(wrap(doubles))
            objective.value
        })
    }

    override fun gradient() : ObjectiveFunctionGradient {
        return ObjectiveFunctionGradient({doubles ->
            setState(wrap(doubles))
            DoubleArray(state.size, {
                index ->
                objective.dualNumber.partialDerivatives.withRespectTo(state[index])
            })
        })
    }

    fun wrap(doubles : DoubleArray) : DoubleArray {
        return DoubleArray(doubles.size, {i ->
            1.0 - Math.abs(1.0 - Math.abs(doubles[i]).rem(2.0))
        })
    }

}