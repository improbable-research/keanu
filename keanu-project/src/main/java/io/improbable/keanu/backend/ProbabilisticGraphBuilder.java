package io.improbable.keanu.backend;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;

public interface ProbabilisticGraphBuilder<T extends ProbabilisticGraph> {

    void convert(Collection<? extends Vertex<?>> vertices);

    void convert(Vertex<?> vertex);

    void alias(VariableReference from, VariableReference to);

    VariableReference add(VariableReference left, VariableReference right);

    void logProb(VariableReference logProbResult);

    void logLikelihood(VariableReference logLikelihoodResult);

    T build();
}
