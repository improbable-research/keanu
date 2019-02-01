package io.improbable.keanu.algorithms.mcmc.testcases;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;

public interface MCMCTestCase {

    BayesianNetwork getModel();

    void assertExpected(NetworkSamples samples);
}
