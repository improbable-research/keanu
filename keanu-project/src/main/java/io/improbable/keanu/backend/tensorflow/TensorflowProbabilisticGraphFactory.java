package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraphConverter;
import io.improbable.keanu.network.BayesianNetwork;


public class TensorflowProbabilisticGraphFactory {

    public static TensorflowProbabilisticGraph convert(BayesianNetwork network) {
        return ProbabilisticGraphConverter.convert(network, new TensorflowProbabilisticGraphBuilder());
    }

}
