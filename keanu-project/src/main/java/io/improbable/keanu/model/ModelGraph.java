package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

public interface ModelGraph<INPUT, OUTPUT> {
    BayesianNetwork getBayesianNetwork();

    void observeValues(INPUT input, OUTPUT output);
}
