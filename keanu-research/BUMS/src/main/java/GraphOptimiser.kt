import io.improbable.keanu.vertices.dbl.DoubleVertex
import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.SimpleValueChecker
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunctionGradient
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer

open class GraphOptimiser(val state : Array<DoubleVertex>, val objective : DoubleVertex) {

    fun setState(doubles : DoubleArray) {
        doubles.forEachIndexed { i, v ->
            state[i].value = v
        }
        objective.lazyEval()
    }


    fun minimise() {
        optimise(GoalType.MINIMIZE, 4000)
    }

    fun maximise() {
        optimise(GoalType.MAXIMIZE, 4000)
    }

    fun optimise(goal : GoalType, maxEvals: Int) {
        val optimizer = NonLinearConjugateGradientOptimizer(NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
                SimpleValueChecker(1e-11, 1e-11))

        val startPoint = DoubleArray(state.size, { i -> state[i].value })

        val optimal = optimizer.optimize(
                MaxEval(maxEvals),
                fitness(),
                gradient(),
                goal,
                InitialGuess(startPoint)
        )
        setState(optimal.point)
    }

    open fun fitness() : ObjectiveFunction {
        return ObjectiveFunction({doubles ->
            setState(doubles)
            objective.value
        })
    }

    open fun gradient() : ObjectiveFunctionGradient {
        return ObjectiveFunctionGradient({doubles ->
            setState(doubles)
            DoubleArray(state.size, {
                index ->
                objective.dualNumber.partialDerivatives.withRespectTo(state[index])
            })
        })
    }

}