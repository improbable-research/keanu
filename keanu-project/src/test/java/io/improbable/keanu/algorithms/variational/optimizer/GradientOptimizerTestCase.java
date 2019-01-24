package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.BayesianNetwork;

public interface GradientOptimizerTestCase {

    BayesianNetwork getModel();

    void assertMLE(OptimizedResult result);

    void assertMAP(OptimizedResult result);
}
