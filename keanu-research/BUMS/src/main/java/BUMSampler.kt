import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D
import java.lang.Math.abs
import java.util.*

class BUMSampler {
    val model3d = Thermometers()
    val model2d = Thermometer2D()
    val modelSphere = ThermometerSphere()
    val optimiser3d : GraphOptimiser
    val optimiser2d : GraphOptimiser
    var random = Random()


    constructor() {
        optimiser3d = GraphOptimiser(arrayOf(model3d.u1, model3d.u2, model3d.u3), model3d.err)
        optimiser2d = GraphOptimiser(arrayOf(model2d.v1,model2d.v2), model2d.thermometers.err)

        model3d.sample()
        optimiser3d.minimise()
        modelSphere.setState(model3d.getState())
    }


    fun samplePerpendicular() {
        var g = random.nextGaussian() * 0.01
        var dl = 0.0001 // step size
        if(g<0) dl = -dl
        var tangentPoint : Vector3D
        while(abs(g)>0.0) {
            val dE1 = Vector3D(
                    model3d.thermo1err.dualNumber.partialDerivatives.withRespectTo(model3d.u1),
                    model3d.thermo1err.dualNumber.partialDerivatives.withRespectTo(model3d.u2),
                    model3d.thermo1err.dualNumber.partialDerivatives.withRespectTo(model3d.u3)
            )
            val dE2 = Vector3D(
                    model3d.thermo2err.dualNumber.partialDerivatives.withRespectTo(model3d.u1),
                    model3d.thermo2err.dualNumber.partialDerivatives.withRespectTo(model3d.u2),
                    model3d.thermo2err.dualNumber.partialDerivatives.withRespectTo(model3d.u3)
            )
            val tangent = dE1.normalize().crossProduct(dE2.normalize())
            if(abs(g) < abs(dl)) dl = g
            tangentPoint = model3d.getState().add(tangent.scalarMultiply(dl))
            if(isOutOfBounds(tangentPoint)) {
                tangentPoint = applyBoundaryConditions(tangentPoint)
                dl = -dl
                g = -g
            }
            val basis = DifferentialGeometry.getBasis(tangent)
            model2d.setBasis(basis[1], basis[2])
            model2d.setOrigin(tangentPoint)
            optimiser2d.minimise()
            model3d.setState(model2d.thermometers.getState())
            g -= dl
        }
    }

    fun sample() {
        var g = random.nextGaussian() * 0.02
        walk(g)
    }

    fun walk(dist : Double) {
        var substeps = 48
        val opt = UnitHypercubeOptimiser(arrayOf(modelSphere.u1, modelSphere.u2, modelSphere.u3),modelSphere.err)
        var dl = dist/substeps // step size
        var tangentPoint : Vector3D
        while(substeps-- > 0) {
            val dE1 = Vector3D(
                    modelSphere.thermo1err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u1),
                    modelSphere.thermo1err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u2),
                    modelSphere.thermo1err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u3)
            )
            val dE2 = Vector3D(
                    modelSphere.thermo2err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u1),
                    modelSphere.thermo2err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u2),
                    modelSphere.thermo2err.dualNumber.partialDerivatives.withRespectTo(modelSphere.u3)
            )
            val tangent = dE1.normalize().crossProduct(dE2.normalize())
            tangentPoint = modelSphere.getState().add(tangent.scalarMultiply(dl))
            if(isOutOfBounds(tangentPoint)) {
                tangentPoint = applyBoundaryConditions(tangentPoint)
                dl = -dl
                modelSphere.setRadius(modelSphere.getState().subtract(tangentPoint).norm)
            } else {
                modelSphere.setRadius(abs(dl))
            }
            modelSphere.setOrigin(modelSphere.getState())
            modelSphere.setState(tangentPoint)
            opt.minimise()
            modelSphere.err.lazyEval()
//            println("${g} ${modelSphere.distFromOrigin() - abs(dl)}")
        }
    }

    fun applyBoundaryConditions(v: Vector3D) : Vector3D {
        return Vector3D(reflectiveBoundary(v.x), reflectiveBoundary(v.y), reflectiveBoundary(v.z))
    }

    fun reflectiveBoundary(r : Double) : Double {
        if(r > 1.0) return(2.0-r)
        if(r < 0.0) return(-r)
        return r
    }

    fun isOutOfBounds(v : Vector3D) : Boolean {
        return isOutOfBounds(v.x) || isOutOfBounds(v.y) || isOutOfBounds(v.z)
    }

    fun isOutOfBounds(r : Double) : Boolean {
        return r>1.0 || r<0.0
    }

}