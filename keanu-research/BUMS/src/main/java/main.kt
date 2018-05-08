import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.SimpleValueChecker
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.gradient.NonLinearConjugateGradientOptimizer
import java.io.FileWriter

fun main(args : Array<String>) {
    val thermo = Thermometers()
    printManifold()
//    val writeToFile = true
//
//    var file :FileWriter? = null
//    if(writeToFile) file = FileWriter("data.out")
//
//    val sampler = BUMSampler()
//    for(i in 1..40000) {
//        sampler.sample()
//        file?.write("${sampler.model3d.temp.value}\n")
//        println("${sampler.model3d.temp.value}")
//    }
//    file?.close()
}


fun printManifold() {
    val model = Thermometers()
    model.sample()
    val opt = GraphOptimiser(arrayOf(model.u2, model.u3), model.err)
   for(T in (170..919).map({i -> i/1000.0})) {
       model.u1.value = T
       model.err.lazyEval()
       opt.minimise()
       println("${model.u1.value} ${model.u2.value} ${model.u3.value}")
    }
}
