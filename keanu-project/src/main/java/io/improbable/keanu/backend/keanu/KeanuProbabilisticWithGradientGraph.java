package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class KeanuProbabilisticWithGradientGraph extends KeanuProbabilisticGraph implements ProbabilisticWithGradientGraph {

    private LogProbGradientCalculator gradientCalculator;
    private Map<VertexId, VertexLabel> idToLabelLookup;

    public KeanuProbabilisticWithGradientGraph(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);
        List<Vertex<DoubleTensor>> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();
        this.idToLabelLookup = continuousLatentVertices.stream().collect(Collectors.toMap(Vertex::getId, Vertex::getLabel));
        this.gradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            continuousLatentVertices
        );
    }

    @Override
    public Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs) {

        if (inputs != null && !inputs.isEmpty()) {
            cascadeUpdate(inputs);
        }

        Map<VertexId, DoubleTensor> gradients = gradientCalculator.getJointLogProbGradientWrtLatents();

        return gradients.entrySet().stream()
            .collect(Collectors.toMap(
                e -> idToLabelLookup.get(e.getKey()).toString(),
                Map.Entry::getValue)
            );
    }

}
