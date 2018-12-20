package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.backend.ProbabilisticWithGradientGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticWithGradientGraph extends KeanuProbabilisticGraph implements ProbabilisticWithGradientGraph {

    private LogProbGradientCalculator gradientCalculator;
    private Map<VertexId, VariableReference> idToLabelLookup;

    public KeanuProbabilisticWithGradientGraph(BayesianNetwork bayesianNetwork) {
        super(bayesianNetwork);
        List<Vertex<DoubleTensor>> continuousLatentVertices = bayesianNetwork.getContinuousLatentVertices();
        this.idToLabelLookup = continuousLatentVertices.stream()
            .collect(toMap(
                Vertex::getId,
                Vertex::getReference
                )
            );
        this.gradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getLatentOrObservedVertices(),
            continuousLatentVertices
        );
    }

    @Override
    public Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs) {

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
