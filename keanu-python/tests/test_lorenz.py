import pandas as pd
import numpy as np
import math
import keanu as kn
from keanu.const import Const
from lorenz_model import LorenzModel


convergedError = 0.01
windowSize = 8
maxWindows = 100

sigma = 10.
beta = 2.66667
rho = 28.
time_step = 0.01

def test_lorenz():
    error = math.inf
    window = 0
    priorMu = (3.,3.,3.)

    model = LorenzModel(sigma, beta, rho, time_step)
    observed = list(model.runModel(windowSize * maxWindows))


    while error > convergedError and window < maxWindows:
        xt0 = kn.Gaussian(priorMu[0], 1.0)
        yt0 = kn.Gaussian(priorMu[1], 1.0)
        zt0 = kn.Gaussian(priorMu[2], 1.0)
        graphTimeSteps = list(buildGraph((xt0, yt0, zt0)))
        xt0.setAndCascade(priorMu[0])
        yt0.setAndCascade(priorMu[1])
        zt0.setAndCascade(priorMu[2])
        applyObservations(graphTimeSteps, window, observed)
        
        optimizer = kn.GradientOptimizer(xt0)
        optimizer.max_a_posteriori()
        posterior = getTimeSliceValues(graphTimeSteps, windowSize - 1)
        
        postT = (window + 1) * (windowSize - 1)
        actualAtPostT = observed[postT]
        
        error = math.sqrt(
                    (actualAtPostT.x - posterior[0].scalar()) ** 2 +
                    (actualAtPostT.y - posterior[1].scalar()) ** 2 +
                    (actualAtPostT.z - posterior[2].scalar()) ** 2
                )
        priorMu = (posterior[0].scalar(), posterior[1].scalar(), posterior[2].scalar())
        window += 1
        
    assert error <= convergedError

def add_time(current):
    rhov = Const(rho)
    (xt, yt, zt) = current

    x_tplus1 = xt * Const(1. - time_step * sigma) + (yt * Const(time_step * sigma))
    y_tplus1 = yt * Const(1. - time_step) + (xt * (rhov - zt) * Const(time_step))
    z_tplus1 = zt * Const(1. - time_step * beta) + (xt * yt * Const(time_step))
    return (x_tplus1, y_tplus1, z_tplus1) 

def buildGraph(initial):
    (x, y, z) = initial
    for _ in range(windowSize):
        yield (x, y, z)
        (x, y, z) = add_time((x, y, z))
    
def applyObservations(graphTimeSteps, window, observed):
    for i in range(len(graphTimeSteps)):
        t = window * (windowSize - 1) + i
        timeSlice = graphTimeSteps[i]
        xt = timeSlice[0]
        observedXt = kn.Gaussian(xt, 1.0)
        observedXt.observe(observed[t].x)

def getTimeSliceValues(timeSteps, time):
    slice = timeSteps[time]
    return list(map(lambda v : v.getValue(), slice))
