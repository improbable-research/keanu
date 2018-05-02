import io.improbable.keanu.kotlin.log
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex

fun logistic(c : DoubleVertex, mean: Double, scale: Double) : DoubleVertex {
    return logistic(c, ConstantDoubleVertex(mean), ConstantDoubleVertex(scale))
}


fun logistic(c : DoubleVertex, mean: DoubleVertex, scale: Double) : DoubleVertex {
    return logistic(c,mean, ConstantDoubleVertex(scale))
}

fun logistic(c : DoubleVertex, mean: DoubleVertex, scale: DoubleVertex) : DoubleVertex {
    val x = -log(ConstantDoubleVertex(1.0) /c - 1.0)
    return x*scale + mean
}
