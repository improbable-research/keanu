package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.BayesianNetwork;

public interface OptimizerTestCase {

    BayesianNetwork getModel();

    void assertMLE(OptimizedResult result);

    void assertMAP(OptimizedResult result);
}
