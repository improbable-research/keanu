package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraphBuilder;
import io.improbable.keanu.backend.Variable;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TensorflowProbabilisticGraphBuilder implements ProbabilisticGraphBuilder<TensorflowProbabilisticGraph> {

    private final TensorflowGraphBuilder graphBuilder;

    private VariableReference logProbResult;
    private VariableReference logLikelihoodResult;

    public TensorflowProbabilisticGraphBuilder() {
        graphBuilder = new TensorflowGraphBuilder();
    }

    @Override
    public void convert(Collection<? extends Vertex<?>> vertices) {
        graphBuilder.convert(vertices);
    }

    @Override
    public void connect(Map<Vertex<?>, Vertex<?>> connections) {
        graphBuilder.connect(connections);
    }

    @Override
    public VariableReference add(VariableReference left, VariableReference right) {
        return graphBuilder.add(left, right);
    }

    @Override
    public void logProb(VariableReference logProbResult) {
        this.logProbResult = logProbResult;
    }

    @Override
    public void logLikelihood(VariableReference logLikelihoodResult) {
        this.logLikelihoodResult = logLikelihoodResult;
    }

    @Override
    public TensorflowProbabilisticGraph build() {

        TensorflowComputableGraph computableGraph = graphBuilder.build();

        List<Variable<?>> latentVariables = graphBuilder.getLatentVariables().stream()
            .map(v -> new TensorflowVariable<>(computableGraph, v))
            .collect(Collectors.toList());

        return new TensorflowProbabilisticGraph(computableGraph, latentVariables, logProbResult, logLikelihoodResult);
    }
}
