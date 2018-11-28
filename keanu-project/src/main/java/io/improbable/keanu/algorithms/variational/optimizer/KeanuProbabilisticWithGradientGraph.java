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
    private Map<VertexId, String> idToLabelLookup;

    public KeanuProbabilisticWithGradientGraph(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);

        List<Vertex<DoubleTensor>> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();

        this.idToLabelLookup = continuousLatentVertices.stream()
            .collect(toMap(
                Vertex::getId,
                KeanuProbabilisticGraph::getUniqueStringReference
                )
            );

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
    public Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs) {
        return gradients(inputs, logProbGradientCalculator);
    }

    @Override
    public Map<String, DoubleTensor> logProbGradients() {
        return logProbGradients(null);
    }

    @Override
    public Map<String, DoubleTensor> logLikelihoodGradients(Map<String, ?> inputs) {
        return gradients(inputs, logLikelihoodGradientCalculator);
    }

    @Override
    public Map<String, DoubleTensor> logLikelihoodGradients() {
        return logLikelihoodGradients(null);
    }

    private Map<String, DoubleTensor> gradients(Map<String, ?> inputs, LogProbGradientCalculator gradientCalculator) {
        if (inputs != null && !inputs.isEmpty()) {
            cascadeUpdate(inputs);
        }

        Map<VertexId, DoubleTensor> gradients = gradientCalculator.getJointLogProbGradientWrtLatents();

        return gradients.entrySet().stream()
            .collect(toMap(
                e -> idToLabelLookup.get(e.getKey()),
                Map.Entry::getValue)
            );
    }

}
