import pandas as pd
import numpy as np
import math
import keanu as kn
from keanu.const import Const
from examples.lorenz_model import LorenzModel


converged_error = 0.01
window_size = 8
max_windows = 100

sigma = 10.
beta = 2.66667
rho = 28.
time_step = 0.01


def test_lorenz():
    error = math.inf
    window = 0
    prior_mu = (3. ,3. ,3.)

    model = LorenzModel(sigma, beta, rho, time_step)
    observed = list(model.run_model(window_size * max_windows))

    while error > converged_error and window < max_windows:
        xt0 = kn.Gaussian(prior_mu[0], 1.0)
        yt0 = kn.Gaussian(prior_mu[1], 1.0)
        zt0 = kn.Gaussian(prior_mu[2], 1.0)
        graph_time_steps = list(build_graph((xt0, yt0, zt0)))
        xt0.setAndCascade(prior_mu[0])
        yt0.setAndCascade(prior_mu[1])
        zt0.setAndCascade(prior_mu[2])
        apply_observations(graph_time_steps, window, observed)
        
        optimizer = kn.GradientOptimizer(xt0)
        optimizer.max_a_posteriori()
        posterior = get_time_slice_values(graph_time_steps, window_size - 1)
        
        post_t = (window + 1) * (window_size - 1)
        actual_at_post_t = observed[post_t]
        
        error = math.sqrt(
                    (actual_at_post_t.x - posterior[0]) ** 2 +
                    (actual_at_post_t.y - posterior[1]) ** 2 +
                    (actual_at_post_t.z - posterior[2]) ** 2
                )
        prior_mu = (posterior[0], posterior[1], posterior[2])
        window += 1
        
    assert error <= converged_error

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
    
def apply_observations(graph_time_steps, window, observed):
    for (idx, time_slice) in enumerate(graph_time_steps):
        t = window * (window_size - 1) + idx
        xt = time_slice[0]
        observed_xt = kn.Gaussian(xt, 1.0)
        observed_xt.observe(observed[t].x)

def get_time_slice_values(time_steps, time):
    time_slice = time_steps[time]
    return list(map(lambda v : v.getValue(), time_slice))
