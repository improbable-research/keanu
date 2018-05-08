import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D
import java.util.*

class BUMSampler {
    val model3d = Thermometers()
    val model2d = Thermometer2D()
    val optimiser3d : GraphOptimiser
    val optimiser2d : GraphOptimiser
    var random = Random()


    constructor() {
        optimiser3d = GraphOptimiser(arrayOf(model3d.u1, model3d.u2, model3d.u3), model3d.err)
        optimiser2d = GraphOptimiser(arrayOf(model2d.v1,model2d.v2), model2d.thermometers.err)

        model3d.sample()
        optimiser3d.minimise()
    }


    fun sample() {
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
        val g = random.nextGaussian() * 0.01
//        var g = 0.005
//        if(random.nextBoolean()) g = -g
        var tangentPoint = model3d.getState().add(tangent.scalarMultiply(g))
        model3d.setState(tangentPoint)
        tangentPoint = model3d.getState()
        val basis = DifferentialGeometry.getBasis(tangentPoint)
        model2d.setBasis(basis[1], basis[2])
        model2d.setOrigin(tangentPoint)
        optimiser2d.minimise()
        model3d.setState(model2d.thermometers.getState())
    }
}