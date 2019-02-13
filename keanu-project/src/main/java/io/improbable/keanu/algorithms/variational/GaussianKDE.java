package io.improbable.keanu.algorithms.variational;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class GaussianKDE {

    public static KDEVertex approximate(Samples<DoubleTensor> vertexSamples) {

        List<Double> samples = vertexSamples.asList().stream()
            .map(GaussianKDE::checkIfScalar)
            .map(tensor -> tensor.scalar())
            .collect(Collectors.toList());

        return new KDEVertex(samples);
    }

    public static KDEVertex approximate(DoubleVertex vertex, Integer nSamples) {
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(vertex.getConnectedGraph());
        DoubleVertexSamples vertexSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(model, ImmutableList.of(vertex), nSamples).getDoubleTensorSamples(vertex);
        return approximate(vertexSamples);
    }

    private static DoubleTensor checkIfScalar(DoubleTensor tensor) throws IllegalArgumentException {
        if (!tensor.isScalar()) {
            throw new IllegalArgumentException("The provided samples are not scalars, but have shape " + Arrays.toString(tensor.getShape()));
        }
        return tensor;
    }
}
