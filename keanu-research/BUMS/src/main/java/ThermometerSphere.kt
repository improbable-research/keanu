import io.improbable.keanu.kotlin.pow
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D
import java.lang.Math.sqrt

class ThermometerSphere : Thermometers {
    val radius = ConstantDoubleVertex(0.0)
    val origin = Array(3,{j -> ConstantDoubleVertex(0.0) })


    constructor() : this(
            UniformVertex(0.0, 1.0),
            UniformVertex(0.0, 1.0),
            UniformVertex(0.0, 1.0)
    )


    constructor(u1 : DoubleVertex, u2 : DoubleVertex, u3 : DoubleVertex) : super(u1,u2,u3) {
        val dx = origin[0] - this.u1
        val dy = origin[1] - this.u2
        val dz = origin[2] - this.u3
        val radiusErr =pow(dx*dx + dy*dy + dz*dz, 0.5) - radius
        err = err + radiusErr*radiusErr
    }


    fun setOrigin(o : Vector3D) {
        origin[0].value = o.x
        origin[1].value = o.y
        origin[2].value = o.z
        err.lazyEval()
    }


    fun setRadius(r : Double) {
        radius.value = r
        err.lazyEval()
    }

    fun distFromOrigin() : Double {
        val dx = u1.value - origin[0].value
        val dy = u2.value - origin[1].value
        val dz = u3.value - origin[2].value
        return sqrt(dx*dx + dy*dy + dz*dz)
    }
}