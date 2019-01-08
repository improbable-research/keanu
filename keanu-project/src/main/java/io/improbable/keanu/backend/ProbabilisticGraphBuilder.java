package io.improbable.keanu.backend;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Map;

public interface ProbabilisticGraphBuilder<T extends ProbabilisticGraph> {

    void convert(Collection<? extends Vertex> vertices);

    void connect(Map<Vertex<?>, Vertex<?>> connections);

    VariableReference add(VariableReference left, VariableReference right);

    void logProb(VariableReference logProbResult);

    void logLikelihood(VariableReference logLikelihoodResult);

    T build();

    default void convert(BayesianNetwork network) {
        ProbabilisticGraphConverter.convert(network, this);
    }

}
