package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.network.BayesianNetwork;


public class KeanuGraphConverter {

    public static KeanuProbabilisticGraph convert(BayesianNetwork network) {
        return new KeanuProbabilisticGraph(network);
    }

    public static KeanuProbabilisticWithGradientGraph convertWithGradient(BayesianNetwork bayesianNetwork) {
        return new KeanuProbabilisticWithGradientGraph(bayesianNetwork);
    }
}
