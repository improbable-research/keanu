import com.sun.org.apache.xpath.internal.operations.Bool
import java.io.FileWriter
import java.lang.Math.abs

fun main(args : Array<String>) {
//    printManifold()
//    walkManifold()
    BUMSample(true)
}

fun BUMSample(writeToFile : Boolean) {

    var file :FileWriter? = null
    if(writeToFile) file = FileWriter("data.out")

    val sampler = BUMSampler()
    for(i in 1..5000) {
        sampler.sample()
        sampler.modelSphere.temp.lazyEval()
        file?.write("${sampler.modelSphere.temp.value}\n")
        println("${sampler.modelSphere.temp.value}")
    }
    file?.close()

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

fun walkManifold() {
    val sampler = BUMSampler()
    sampler.modelSphere.sample()
    val opt = GraphOptimiser(arrayOf(sampler.modelSphere.u2, sampler.modelSphere.u3), sampler.modelSphere.err)
    sampler.modelSphere.u1.value = 0.25
    sampler.modelSphere.err.lazyEval()
    opt.minimise()
    println("${sampler.modelSphere.u1.value} ${sampler.modelSphere.u2.value} ${sampler.modelSphere.u3.value}")
    for(i in 1..500) {
        sampler.walk(0.005)
        sampler.modelSphere.err.lazyEval()
//        println("${sampler.modelSphere.u1.value} ${sampler.modelSphere.u2.value} ${sampler.modelSphere.u3.value} ${sampler.modelSphere.err.value}")
//        println("${sampler.modelSphere.u1.value} ${sampler.modelSphere.u2.value} ${sampler.modelSphere.u3.value}")
        println("${sampler.modelSphere.temp.value}")
    }
}
