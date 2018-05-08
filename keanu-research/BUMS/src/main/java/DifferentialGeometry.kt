import org.apache.commons.math3.geometry.euclidean.threed.Vector3D

object DifferentialGeometry {
    fun getBasis(vec: Vector3D) : Array<Vector3D> {
        val b0 = vec.normalize()
        var b1 = b0.crossProduct(Vector3D(1.0,0.0,0.0))
        if(b1.norm == 0.0) b1 = Vector3D(0.0,1.0,0.0)
        val b2 = b0.crossProduct(b1)
        return arrayOf(b0,b1,b2)
    }

}