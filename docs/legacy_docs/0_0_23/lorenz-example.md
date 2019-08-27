---
# Page settings
layout: default
keywords: lorenz example
comments: false
version: 0.0.23
permalink: /docs/0_0_23/examples/lorenz/

# Hero section
title: Lorenz Example
description: A more complex example with a chaotic system

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_23/examples/thermometer/'
---

## Overview

This example aims to demonstrate two things:

* using Keanu with [Kotlin](https://kotlinlang.org/)
* how to solve a time-stepped problem

## The problem to model

Imagine you are watching the point below moving through space:

![Trajectory through phase space](https://upload.wikimedia.org/wikipedia/commons/1/13/A_Trajectory_Through_Phase_Space_in_a_Lorenz_Attractor.gif)

You'd begin to notice it has an interesting repeated pattern of movement. If you had a really
keen eye, you'd notice that its movement can be described by the
[Lorenz equations](https://en.wikipedia.org/wiki/Lorenz_system).

The Lorenz equations are particuarly hard to model as they describe a chaotic system. Two initial
starting states, no matter how close, will diverge almost immediately. 

In this example, you'll make observations on the point above and build a probabilistic model
using Keanu that can accurately describe its chaotic motion. To make things harder, you are going to only observe 
the X coordinate, and those observations will be noisy.

### How to model the problem

To start with, you'll need to build a simple program that can step through time and calculate
coordinates using the Lorenz equations. This will serve as the “Real World” system from which you can take observations 
of the X coordinate, and against which you can compare the calculated coordinates of the probabilistic model.

Then you'll step through time, taking observations of the X coordinate from this "Real World" system, 
and feeding this value into the probabilistic model.

What does this probabilistic model look like? Fortunately, you already have a conceptual model of the
system and you know it follows movement described by the Lorenz equations: 

![Lorenz equations](https://wikimedia.org/api/rest_v1/media/math/render/svg/5f993e17e16f1c3ea4ad7031353c61164a226bb8)

Rather than performing a single timestep and then calculating the most likely value of the coordinates, 
you'll perform several timesteps at once and then calculate the most likely value. 

Let's define 10 timesteps as a window, and calculate the probable values at the end of every window.

## Pseudocode

At each timestep, you need to:

1. Increment the "Real World" system and take a reading of its coordinates.
2. Create a probabilistic graph that describes the Lorenz equations for that timestep.
3. Observe the X coordinate of this graph to be the X coordinate from the "Real World" system.

The pseudocode for a window looks like:

1. Run 10 timesteps
1. Calculate the most probable value of the X, Y and Z coordinates from the connected graph.
2. Use these probable values as a prior value of X, Y and Z in the graph for the next window.
3. Evaluate how close the probable values are to the actual value of the coordinates.

## Implementation

Let's start by calculating the coordinates of the "Real World" system, for a given number of timesteps.
We've prepared this ahead of time, so you can use our class, `LorenzModel`.

```kotlin
val model = LorenzModel()
val lorenzCoordinates = model.runModel(maxWindows * windowSize)
```

Let's define a starting point for the X, Y and Z coordinates. 
You have no prior knowledge of what the starting coordinates for the model will be, so let's define each
coordinate as a Gaussian around the origin.

```kotlin
val random = DoubleVertexFactory()
val origin = 0.0
var initX = random.nextGaussian(origin, 2.5)
var initY = random.nextGaussian(origin, 2.5)
var initZ = random.nextGaussian(origin, 2.5)
```

You then want to iterate through these timesteps until either:

1. The calculation of the most probable values of X, Y and Z are within an acceptable tolerance
of the actual coordinates.
2. We reach our maximum timestep.

Place all of the following code inside a loop that describes these conditons:

```kotlin
while (error > convergedError && window < maxWindows) {
```

Let's now define a function that can calculate the next timestep of a Lorenz equation in Keanu
given these starting coordinates.

Fortunately, Kotlin supports operator overloading, so this will look exactly like a normal
program even though you're dealing with objects representing probability distributions.

```kotlin
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
```

Let's define a function that can compute the 10 timesteps in a window using the function you defined above. 

```kotlin
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
```
 
You can now apply the observations of the X coordinate from the "Real World" system to each timestep
in the window calculated above. 

Let's define a function that does this.

```kotlin
fun applyObservations(graphTimeSteps: MutableList<List<DoubleVertex>>, windowSize: Int, window: Int, observed: List<LorenzModel.Coordinates>, random: DoubleVertexFactory) {
    for (i in graphTimeSteps.indices) {
        val time = window * (windowSize - 1) + i
        val timestep = graphTimeSteps[i]
        val xCoord = timestep.first()
        val observedXCoord = random.nextGaussian(xCoord, 1.0)
        observedXCoord.observe(observed[time].x)
    }
}
```

This function is iterating through each of the timesteps you calculated above, extracting the Gaussian
distribution that describes the X coordinate, and observing it to be the value taken from the
"Real World" system.

Brilliant, now let's calculate the most likely value for the X, Y and Z coordinates.

Hopefully by now the code for performing Gradient Optimisation will look familiar to you:

```kotlin
val net = BayesianNetwork(graphTimeSteps.first().first().connectedGraph)
val optimiser = Optimizer.of(net)
optimiser.maxAPosteriori()
```

Now you've calculated the most probable values for the graph, you need to feed these into the next window.
As you're performing 10 timesteps in a window, you want to take the value of the 10th timestep as the
starting point of the next window.

Let's define a function to do that:

```kotlin
fun getTimestepValues(graphTimeSteps: MutableList<List<DoubleVertex>>, time: Int): List<Double> {
    val timestep = graphTimeSteps[time]
    return timestep.stream().mapToDouble { value -> value.value.scalar() }.toArray().toList()
}
```

Great, now you need to feed these back into the next window.

If you'll recall, at the start of the window, you defined the initial coordinates of the graph as:

```kotlin
var initX = random.nextGaussian(origin, 2.5)
var initY = random.nextGaussian(origin, 2.5)
var initZ = random.nextGaussian(origin, 2.5)
```

Therefore, let's recalculate these Gaussians to include the calculated value of the most probable coordinates:

```kotlin
val posterior = getTimestepValues(graphTimeSteps, windowSize - 1)
val postTimestep = (window + 1) * (windowSize - 1)
val coordinatesAtPostTimestep = lorenzCoordinates[postTimestep]

error = error(coordinatesAtPostTimestep, posterior)
println("Error: " + error)

initX = random.nextGaussian(posterior[0], 2.5)
initY = random.nextGaussian(posterior[1], 2.5)
initZ = random.nextGaussian(posterior[2], 2.5)
```

## Results

Can you have a guess at what you think the results will be before you run the code?

We are performing 10 timesteps in each window: how many windows do you think it will take before the
error between the probabilistic model and the "Real World" model is negligible?

Printing out the error of each window produces the following results.

```
Error: 4.528976372510882
Error: 3.72098313476314
Error: 2.665636788694244
Error: 1.812040602844943
Error: 1.2950274565457909
Error: 1.040466635607921
Error: 0.7338159078803722
Error: 0.38109769602303945
Error: 0.1774717811757308
Error: 0.11441697844622131
Error: 0.0918083820220631
Error: 0.06013662151356879
Error: 0.030568642144244644
Error: 0.02316773884356288
Error: 0.015850612891648755
Error: 0.008663664263278796
```

It takes 16 windows before the probabilistic model has converged on the "Real World" system, from only an
observation of the X coordinate!

## Code

Here is the completed code if you'd like to run it yourself.

```kotlin
private val windowSize = 10
private val maxWindows = 100
private val convergedError = 0.01
private var window = 0
private var error = Double.MAX_VALUE

fun main(args: Array<String>) {

    val model = LorenzModel()
    val lorenzCoordinates = model.runModel(maxWindows * windowSize)

    val random = DoubleVertexFactory()
    val origin = 0.0
    var initX = random.nextGaussian(origin, 2.5)
    var initY = random.nextGaussian(origin, 2.5)
    var initZ = random.nextGaussian(origin, 2.5)

    while (error > convergedError && window < maxWindows) {

        val initialConditions = listOf(initX, initY, initZ)
        val graphTimeSteps = mutableListOf(initialConditions) as MutableList<List<DoubleVertex>>

        calculateLorenzTimesteps(graphTimeSteps, windowSize)

        applyObservations(graphTimeSteps, windowSize, window, lorenzCoordinates, random)

        val net = BayesianNetwork(graphTimeSteps.first().first().connectedGraph)
        val optimiser = Optimizer.of(net)
        optimiser.maxAPosteriori()

        val posterior = getTimestepValues(graphTimeSteps, windowSize - 1)
        val postTimestep = (window + 1) * (windowSize - 1)
        val coordinatesAtPostTimestep = lorenzCoordinates[postTimestep]

        error = error(coordinatesAtPostTimestep, posterior)
        println("Error: " + error)

        initX = random.nextGaussian(posterior[0], 2.5)
        initY = random.nextGaussian(posterior[1], 2.5)
        initZ = random.nextGaussian(posterior[2], 2.5)

        window++
    }
}

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

fun applyObservations(graphTimeSteps: MutableList<List<DoubleVertex>>, windowSize: Int, window: Int, observed: List<LorenzModel.Coordinates>, random: DoubleVertexFactory) {
    for (i in graphTimeSteps.indices) {
        val time = window * (windowSize - 1) + i
        val timestep = graphTimeSteps[i]
        val xCoord = timestep.first()
        val observedXCoord = random.nextGaussian(xCoord, 1.0)
        observedXCoord.observe(observed[time].x)
    }
}

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

fun getTimestepValues(graphTimeSteps: MutableList<List<DoubleVertex>>, time: Int): List<Double> {
    val timestep = graphTimeSteps[time]
    return timestep.stream().mapToDouble { value -> value.value.scalar() }.toArray().toList()
}

fun error(coordinates: LorenzModel.Coordinates, posterior: List<Double>): Double {
    return Math.sqrt(
        Math.pow(coordinates.x - posterior[0], 2.0) +
            Math.pow(coordinates.y - posterior[1], 2.0) +
            Math.pow(coordinates.z - posterior[2], 2.0)
    )
}
```

## Java Code

Here is the same implementation, but done in Java.

```java
public class LorenzTest {

    @Ignore
    public void convergesOnLorenz() {

        double[] priorMu = new double[]{3, 3, 3};
        double error = Double.MAX_VALUE;
        double convergedError = 0.01;
        int windowSize = 8;
        int window = 0;
        int maxWindows = 100;

        LorenzModel model = new LorenzModel();
        List<LorenzModel.Coordinates> observed = model.runModel(windowSize * maxWindows);

        while (error > convergedError && window < maxWindows) {

            GaussianVertex xt0 = new GaussianVertex(priorMu[0], 1.0);
            GaussianVertex yt0 = new GaussianVertex(priorMu[1], 1.0);
            GaussianVertex zt0 = new GaussianVertex(priorMu[2], 1.0);

            List<List<DoubleVertex>> graphTimeSteps = new ArrayList<>();
            graphTimeSteps.add(Arrays.asList(xt0, yt0, zt0));

            //Build graph
            for (int i = 1; i < windowSize; i++) {
                List<DoubleVertex> ti = graphTimeSteps.get(i - 1);
                List<DoubleVertex> tiPlus1 = addTime(
                    ti.get(0), ti.get(1), ti.get(2),
                    LorenzModel.timeStep, LorenzModel.sigma, LorenzModel.rho, LorenzModel.beta
                );
                graphTimeSteps.add(tiPlus1);
            }

            xt0.setAndCascade(priorMu[0]);
            yt0.setAndCascade(priorMu[1]);
            zt0.setAndCascade(priorMu[2]);

            //Apply observations
            for (int i = 0; i < graphTimeSteps.size(); i++) {

                int t = window * (windowSize - 1) + i;

                List<DoubleVertex> timeSlice = graphTimeSteps.get(i);
                DoubleVertex xt = timeSlice.get(0);

                GaussianVertex observedXt = new GaussianVertex(xt, 1.0);
                observedXt.observe(observed.get(t).x);
            }

            BayesianNetwork net = new BayesianNetwork(xt0.getConnectedGraph());

            Optimizer graphOptimizer = Keanu.Optimizer.of(net);

            graphOptimizer.maxAPosteriori();

            List<DoubleTensor> posterior = getTimeSliceValues(graphTimeSteps, windowSize - 1);

            int postT = (window + 1) * (windowSize - 1);
            LorenzModel.Coordinates actualAtPostT = observed.get(postT);

            error = Math.sqrt(
                Math.pow(actualAtPostT.x - posterior.get(0).scalar(), 2) +
                    Math.pow(actualAtPostT.y - posterior.get(1).scalar(), 2) +
                    Math.pow(actualAtPostT.z - posterior.get(2).scalar(), 2)
            );

            System.out.println("Error: " + error);

            priorMu = new double[]{posterior.get(0).scalar(), posterior.get(1).scalar(), posterior.get(2).scalar()};
            window++;
        }

        Assert.assertTrue(error <= convergedError);
    }

    private List<DoubleVertex> addTime(DoubleVertex xt,
                                       DoubleVertex yt,
                                       DoubleVertex zt,
                                       double timestep,
                                       double sigma,
                                       double rho,
                                       double beta) {

        DoubleVertex rhov = ConstantVertex.of(rho);

        DoubleVertex xtplus1 = xt.multiply(1 - timestep * sigma).plus(yt.multiply(timestep * sigma));

        DoubleVertex ytplus1 = yt.multiply(1 - timestep).plus(xt.multiply(rhov.minus(zt)).multiply(timestep));

        DoubleVertex ztplus1 = zt.multiply(1 - timestep * beta).plus(xt.multiply(yt).multiply(timestep));

        return Arrays.asList(xtplus1, ytplus1, ztplus1);
    }

    private List<DoubleTensor> getTimeSliceValues(List<List<DoubleVertex>> graphTimeSteps, int time) {
        List<DoubleVertex> slice = graphTimeSteps.get(time);

        return slice.stream()
            .map(Vertex::getValue)
            .collect(Collectors.toList());
    }

}
```

## Python Code

Here is the same implementation, but done in Python.

```java
converged_error = 0.01
window_size = 8
max_windows = 100

sigma = 10.
beta = 2.66667
rho = 28.
time_step = 0.01

Coordinates = namedtuple('Coordinates', 'x y z')

class LorenzModel:

    def __init__(self, sigma, beta, rho, time_step) -> None:
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.time_step = time_step

    def run_model(self, num_time_steps):
        position = Coordinates(2., 5., 4.)
        yield position
        for _ in range(num_time_steps):
            position = self.__get_next_position(position)
            yield position

    def __get_next_position(self, current: Coordinates):
        nextX = current.x + self.time_step * (self.sigma * (current.y - current.x))
        nextY = current.y + self.time_step * (current.x * (self.rho - current.z) - current.y)
        nextZ = current.z + self.time_step * (current.x * current.y - self.beta * current.z)
        return Coordinates(nextX, nextY, nextZ)

def add_time(current):
    rho_v = Const(rho)
    (xt, yt, zt) = current

    x_tplus1 = xt * Const(1. - time_step * sigma) + (yt * Const(time_step * sigma))
    y_tplus1 = yt * Const(1. - time_step) + (xt * (rho_v - zt) * Const(time_step))
    z_tplus1 = zt * Const(1. - time_step * beta) + (xt * yt * Const(time_step))
    return (x_tplus1, y_tplus1, z_tplus1)


def build_graph(initial):
    (x, y, z) = initial
    yield (x, y, z)
    for _ in range(window_size):
        (x, y, z) = add_time((x, y, z))
        yield (x, y, z)


def apply_observations(graph_time_steps, window,
                       observed):
    for (idx, time_slice) in enumerate(graph_time_steps):
        t = window * (window_size - 1) + idx
        xt = time_slice[0]
        observed_xt = Gaussian(xt, 1.0)
        observed_xt.observe(observed[t].x)


def get_time_slice_values(time_steps, time):
    time_slice = time_steps[time]
    return list(map(lambda v: v.get_value(), time_slice))

def test_run_lorenz():
    error = math.inf
    window = 0
    prior_mu = (3., 3., 3.)

    model = LorenzModel(sigma, beta, rho, time_step)
    observed = list(model.run_model(window_size * max_windows))

    while error > converged_error and window < max_windows:
        xt0 = Gaussian(prior_mu[0], 1.0)
        yt0 = Gaussian(prior_mu[1], 1.0)
        zt0 = Gaussian(prior_mu[2], 1.0)
        graph_time_steps = list(build_graph((xt0, yt0, zt0)))
        xt0.set_and_cascade(prior_mu[0])
        yt0.set_and_cascade(prior_mu[1])
        zt0.set_and_cascade(prior_mu[2])
        apply_observations(graph_time_steps, window, observed)

        optimizer = GradientOptimizer(xt0)
        optimizer.max_a_posteriori()
        posterior = get_time_slice_values(graph_time_steps, window_size - 1)

        post_t = (window + 1) * (window_size - 1)
        actual_at_post_t = observed[post_t]

        error = math.sqrt((actual_at_post_t.x - posterior[0]) ** 2 + (actual_at_post_t.y - posterior[1]) ** 2 +
                          (actual_at_post_t.z - posterior[2]) ** 2)
        prior_mu = (posterior[0], posterior[1], posterior[2])
        window += 1
```

