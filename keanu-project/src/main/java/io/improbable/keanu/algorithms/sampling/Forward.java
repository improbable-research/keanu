package io.improbable.keanu.algorithms.sampling;

import com.google.common.collect.Sets;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import org.nd4j.base.Preconditions;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class Forward implements PosteriorSamplingAlgorithm {

    private final KeanuRandom random;

    public Forward() {
        this(KeanuRandom.getDefaultRandom());
    }

    public Forward(KeanuRandom random) {
        this.random = random;
    }

    /**
     * Samples from the prior of a Probabilistic Model.
     * Bayesian Network is the only supported implementation.
     *
     * Samples are taken by sampling from the prior of the desired variables in topological order.
     *
     * @param model the model to sample from
     * @param variablesToSampleFrom the variables to sample from
     * @param sampleCount the number of samples to take
     * @return sampling samples of a computable graph
     */
    @Override
    public NetworkSamples getPosteriorSamples(ProbabilisticModel model, List<? extends Variable> variablesToSampleFrom, int sampleCount) {
        return generatePosteriorSamples(model, variablesToSampleFrom)
            .generate(sampleCount);
    }

    @Override
    public NetworkSamplesGenerator generatePosteriorSamples(ProbabilisticModel model, List<? extends Variable> variablesToSampleFrom) {
        return new NetworkSamplesGenerator(setupSampler(model, variablesToSampleFrom), StatusBar::new);
    }

    private SamplingAlgorithm setupSampler(ProbabilisticModel model, List<? extends Variable> variablesToSampleFrom) {
        List<? extends Variable> latentVariables = model.getLatentVariables();
        Preconditions.checkArgument(latentVariables.size() > 0, "Your model must contain latent variables in order to forward sample.");

        Variable latent = latentVariables.get(0);
        Preconditions.checkArgument(latent instanceof Vertex, "The Forward Sampler only works for Variables of type Vertex.");

        BayesianNetwork network = new BayesianNetwork(((Vertex) latent).getConnectedGraph());

        List<Vertex> observedVertices = network.getObservedVertices();
        assertUpstreamOfObservedDoesNotContainProbabilistic(observedVertices);

        List<Vertex> verticesToSampleFrom = variablesToSampleFrom.stream().map(v -> (Vertex) v).collect(Collectors.toList());
        Set<Vertex> allUpstreamVertices = allUpstreamVertices(verticesToSampleFrom);

        List<Vertex> sortedVertices = TopologicalSort.sort(allUpstreamVertices);
        sortedVertices = removeNonProbabilisticVerticesBeforeTheFirstProbabilistic(sortedVertices);

        return new ForwardSampler(verticesToSampleFrom, sortedVertices, random);
    }

    private List<Vertex> removeNonProbabilisticVerticesBeforeTheFirstProbabilistic(List<Vertex> topologicalVertices) {
        List<Vertex> copy = new ArrayList<>(topologicalVertices);

        for (Vertex vertex : topologicalVertices) {
            if (vertex instanceof Probabilistic) {
                break;
            } else {
                copy.remove(vertex);
            }
        }
        return copy;
    }

    private Set<Vertex> allUpstreamVertices(List<Vertex> fromVertices) {
        Set<Vertex> upstream = new HashSet<>();
        for (Vertex vertex : fromVertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, true);
            Set<Vertex> vertexUpsteam = upstreamLambdaSection.getAllVertices();
            upstream = Sets.union(upstream, vertexUpsteam);
        }
        return upstream;
    }

    private void assertUpstreamOfObservedDoesNotContainProbabilistic(List<Vertex> vertices) {
        for (Vertex vertex : vertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, false);
            Set<Vertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();
            Preconditions.checkArgument(upstreamRandomVariables.size() == 1, "Vertex: [" + vertex + "] has a random variable in its upstream lambda section");
        }
    }
}
