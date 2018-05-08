import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D

class Thermometer2D {
    val basis1 = Array(3,{j -> ConstantDoubleVertex(0.0) })
    val basis2 = Array(3,{j -> ConstantDoubleVertex(0.0) })
    val origin = Array(3,{j -> ConstantDoubleVertex(0.0) })
    val thermometers : Thermometers
    val v1 = ConstantDoubleVertex(0.0)
    val v2 = ConstantDoubleVertex(0.0)

    constructor() {
        val u1 = origin[0] + basis1[0]*v1 + basis2[0]*v2
        val u2 = origin[1] + basis1[1]*v1 + basis2[1]*v2
        val u3 = origin[2] + basis1[2]*v1 + basis2[2]*v2
        thermometers = Thermometers(u1,u2,u3)
    }

    fun setBasis(b1 : Vector3D, b2 : Vector3D) {
        basis1[0].value = b1.x
        basis1[1].value = b1.y
        basis1[2].value = b1.z
        basis2[0].value = b2.x
        basis2[1].value = b2.y
        basis2[2].value = b2.z
        thermometers.err.lazyEval()
    }

    fun setOrigin(o : Vector3D) {
        origin[0].value = o.x
        origin[1].value = o.y
        origin[2].value = o.z
        thermometers.err.lazyEval()
    }

}