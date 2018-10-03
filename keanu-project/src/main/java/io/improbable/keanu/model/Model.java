package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

public interface Model {

    BayesianNetwork buildModel();

    BayesianNetwork addObservationLayer(BayesianNetwork net);

    BayesianNetwork getNet();

}
