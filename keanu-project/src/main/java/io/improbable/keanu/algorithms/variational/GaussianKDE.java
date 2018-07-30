package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class GaussianKDE {

    public KDEVertex approximate(VertexSamples<DoubleTensor> vertexSamples){

        List<Double> samples = vertexSamples.asList().stream()
            .map(tensor -> checkIfScalar(tensor))
            .map(tensor -> tensor.scalar())
            .collect(Collectors.toList());

        return new KDEVertex(samples);
    }

    public KDEVertex approximate(DoubleVertex vertex, Integer nSamples){
        BayesianNetwork network = new BayesianNetwork(vertex.getConnectedGraph());
        DoubleVertexSamples vertexSamples = MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(network, Arrays.asList(vertex), nSamples).getDoubleTensorSamples(vertex);
        return approximate(vertexSamples);
    }

    private DoubleTensor checkIfScalar(DoubleTensor tensor) throws IllegalArgumentException {
        if (tensor.asFlatList().size()>1){
            throw new IllegalArgumentException();
        }
        return tensor;
    }
}
