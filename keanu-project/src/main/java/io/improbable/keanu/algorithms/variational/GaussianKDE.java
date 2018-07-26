package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class GaussianKDE {

    public KDEVertex approximate(VertexSamples<DoubleTensor> vertexSamples){

        List<Double> samples = vertexSamples.asList().stream()
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
}
