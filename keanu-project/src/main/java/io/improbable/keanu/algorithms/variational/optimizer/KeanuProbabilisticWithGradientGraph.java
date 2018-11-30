package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticWithGradientGraph extends KeanuProbabilisticGraph implements ProbabilisticWithGradientGraph {

    private LogProbGradientCalculator logProbGradientCalculator;
    private LogProbGradientCalculator logLikelihoodGradientCalculator;
    private Map<VertexId, VariableReference> idToLabelLookup;

    public KeanuProbabilisticWithGradientGraph(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);

        List<Vertex<DoubleTensor>> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();

        this.idToLabelLookup = continuousLatentVertices.stream()
            .collect(toMap(
                Vertex::getId,
                v -> v
            ));

        this.logProbGradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            continuousLatentVertices
        );

        this.logLikelihoodGradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getObservedVertices(),
            continuousLatentVertices
        );
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logProbGradientCalculator);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs) {
        return gradients(inputs, logLikelihoodGradientCalculator);
    }

    @Override
    public Map<VariableReference, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    private Map<VariableReference, DoubleTensor> gradients(Map<VariableReference, ?> inputs, LogProbGradientCalculator gradientCalculator) {
        if (inputs != null && !inputs.isEmpty()) {
            cascadeUpdate(inputs);
        }

        Map<VertexId, DoubleTensor> gradients = gradientCalculator.getJointLogProbGradientWrtLatents();

        return gradients.entrySet().stream()
            .collect(toMap(
                e -> idToLabelLookup.get(e.getKey()),
                Map.Entry::getValue
            ));
    }

}
