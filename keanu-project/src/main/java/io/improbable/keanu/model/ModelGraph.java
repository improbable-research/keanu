package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

public interface ModelGraph<INPUT, OUTPUT> {
    BayesianNetwork getNet();

    void observeValues(INPUT input, OUTPUT output);

    OUTPUT predict(INPUT input);
}
