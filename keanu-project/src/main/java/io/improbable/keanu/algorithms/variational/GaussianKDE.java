package io.improbable.keanu.algorithms.variational;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class GaussianKDE {

    public static KDEVertex approximate(VertexSamples<DoubleTensor> vertexSamples) {

        List<Double> samples = vertexSamples.asList().stream()
            .map(GaussianKDE::checkIfScalar)
            .map(tensor -> tensor.scalar())
            .collect(Collectors.toList());

        return new KDEVertex(samples);
    }

    public static KDEVertex approximate(DoubleVertex vertex, Integer nSamples) {
        BayesianNetwork network = new BayesianNetwork(vertex.getConnectedGraph());
        ProbabilisticModel graph = new KeanuProbabilisticModel(network);
        DoubleVertexSamples vertexSamples = MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(graph, ImmutableList.of(vertex), nSamples).getDoubleTensorSamples(vertex);
        return approximate(vertexSamples);
    }

    private static DoubleTensor checkIfScalar(DoubleTensor tensor) throws IllegalArgumentException {
        if (!tensor.isScalar()) {
            throw new IllegalArgumentException("The provided samples are not scalars, but have shape " + Arrays.toString(tensor.getShape()));
        }
        return tensor;
    }
}
