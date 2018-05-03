import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.SimpleValueChecker
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
import java.io.FileWriter

fun main(args : Array<String>) {

    val objective = Thermometers()
    val file = FileWriter("data.out")
    for(i in 1..1000) {
        getProposal(objective, i)
        objective.walk()
        file.write("${objective.temp.value}\n")
//        println("${objective.temp.value}")
    }
    file.close()
}

fun getProposal(objective : Thermometers, iteration: Int) {
    val optimizer = NonLinearConjugateGradientOptimizer(
            NonLinearConjugateGradientOptimizer.Formula.POLAK_RIBIERE,
            SimpleValueChecker(1e-5, 1e-6)
    )

    if (iteration == 0) {
        objective.sample()
    }

    val startPoint = doubleArrayOf(objective.u1.value, objective.u2.value, objective.u3.value)

    val optimal = optimizer.optimize(
            MaxEval(4000),
            objective.fitness(),
            objective.gradient(),
            GoalType.MINIMIZE,
            InitialGuess(startPoint)
    )
    objective.setState(optimal.point)
}