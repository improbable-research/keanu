package io.improbable.snippet


import io.improbable.keanu.algorithms.variational.optimizer.Optimizer
import io.improbable.keanu.kotlin.times
import io.improbable.keanu.network.BayesianNetwork
import io.improbable.keanu.randomfactory.DoubleVertexFactory
import io.improbable.keanu.vertices.dbl.DoubleVertex
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex

//%%SNIPPET_START%% LorenzFull
private val windowSize = 10
private val maxWindows = 100
private val convergedError = 0.01
private var window = 0
private var error = Double.MAX_VALUE

fun main(args: Array<String>) {

//%%SNIPPET_START%% LorenzModel
val model = LorenzModel()
val lorenzCoordinates = model.runModel(maxWindows * windowSize)
//%%SNIPPET_END%% LorenzModel

//%%SNIPPET_START%% LorenzStartingPoint
val random = DoubleVertexFactory()
val origin = 0.0
//%%SNIPPET_START%% LorenzStartValues
var initX = random.nextGaussian(origin, 2.5)
var initY = random.nextGaussian(origin, 2.5)
var initZ = random.nextGaussian(origin, 2.5)
//%%SNIPPET_END%% LorenzStartingPoint
//%%SNIPPET_END%% LorenzStartValues

//%%SNIPPET_START%% LorenzIterate
while (error > convergedError && window < maxWindows) {
//%%SNIPPET_END%% LorenzIterate

val initialConditions = listOf(initX, initY, initZ)
val graphTimeSteps = mutableListOf(initialConditions) as MutableList<List<DoubleVertex>>

calculateLorenzTimesteps(graphTimeSteps, windowSize)

applyObservations(graphTimeSteps, windowSize, window, lorenzCoordinates, random)

//%%SNIPPET_START%% LorenzOptimise
val net = BayesianNetwork(graphTimeSteps.first().first().connectedGraph)
val optimiser = Optimizer.of(net)
optimiser.maxAPosteriori()
//%%SNIPPET_END%% LorenzOptimise

//%%SNIPPET_START%% LorenzNewStartValues
val posterior = getTimestepValues(graphTimeSteps, windowSize - 1)
val postTimestep = (window + 1) * (windowSize - 1)
val coordinatesAtPostTimestep = lorenzCoordinates[postTimestep]

error = error(coordinatesAtPostTimestep, posterior)
println("Error: " + error)

initX = random.nextGaussian(posterior[0], 2.5)
initY = random.nextGaussian(posterior[1], 2.5)
initZ = random.nextGaussian(posterior[2], 2.5)
//%%SNIPPET_END%% LorenzNewStartValues

window++
}
}

//%%SNIPPET_START%% LorenzWindow
fun calculateLorenzTimesteps(graphTimeSteps: MutableList<List<DoubleVertex>>, windowSize: Int) {
    for (i in 0.until(windowSize - 1)) {
        val startConditions = graphTimeSteps[i]
        val timesteppedCoordinates = lorenzTimestep(startConditions.first(),
                startConditions[1],
                startConditions.last(),
                LorenzModel.timeStep,
                LorenzModel.sigma,
                LorenzModel.rho,
                LorenzModel.beta)
        graphTimeSteps.add(timesteppedCoordinates)
    }
}
//%%SNIPPET_END%% LorenzWindow

//%%SNIPPET_START%% LorenzObservations
fun applyObservations(graphTimeSteps: MutableList<List<DoubleVertex>>, windowSize: Int, window: Int, observed: List<LorenzModel.Coordinates>, random: DoubleVertexFactory) {
    for (i in graphTimeSteps.indices) {
        val time = window * (windowSize - 1) + i
        val timestep = graphTimeSteps[i]
        val xCoord = timestep.first()
        val observedXCoord = random.nextGaussian(xCoord, 1.0)
        observedXCoord.observe(observed[time].x)
    }
}
//%%SNIPPET_END%% LorenzObservations

//%%SNIPPET_START%% LorenzTimestep
fun lorenzTimestep(xCoord: DoubleVertex, yCoord: DoubleVertex, zCoord: DoubleVertex, timestep: Double, sigma: Double, rho: Double, beta: Double): List<DoubleVertex> {
    val constantRho = ConstantDoubleVertex(rho)
    val deltaX = timestep * sigma * (yCoord - xCoord)
    val xCoordNextTimestep = xCoord + deltaX
    val deltaY = timestep * (xCoord * (constantRho - zCoord) - yCoord)
    val yCoordNextTimestep = yCoord + deltaY
    val deltaZ = timestep * ((xCoord * yCoord) - (beta * zCoord))
    val zCoordNextTimestep = zCoord + deltaZ
    return listOf(xCoordNextTimestep, yCoordNextTimestep, zCoordNextTimestep)
}
//%%SNIPPET_END%% LorenzTimestep

//%%SNIPPET_START%% LorenzGetWindowVals
fun getTimestepValues(graphTimeSteps: MutableList<List<DoubleVertex>>, time: Int): List<Double> {
    val timestep = graphTimeSteps[time]
    return timestep.stream().mapToDouble { value -> value.value.scalar() }.toArray().toList()
}
//%%SNIPPET_END%% LorenzGetWindowVals

fun error(coordinates: LorenzModel.Coordinates, posterior: List<Double>): Double {
    return Math.sqrt(
            Math.pow(coordinates.x - posterior[0], 2.0) +
                    Math.pow(coordinates.y - posterior[1], 2.0) +
                    Math.pow(coordinates.z - posterior[2], 2.0)
    )
}
//%%SNIPPET_END%% LorenzFull