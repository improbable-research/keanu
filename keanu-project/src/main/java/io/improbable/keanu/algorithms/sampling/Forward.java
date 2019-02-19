package io.improbable.keanu.algorithms.sampling;

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
import io.improbable.keanu.network.TransitiveClosure;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.Vertex;
import org.nd4j.base.Preconditions;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.Sets;

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

        for (Variable variable : variablesToSampleFrom) {
            Preconditions.checkArgument(variable instanceof Vertex, "The Forward Sampler only works for Variables of type Vertex. Received : " + variable);
        }

        Variable latent = latentVariables.get(0);
        BayesianNetwork network = new BayesianNetwork(((Vertex) latent).getConnectedGraph());

        List<Vertex> observedVertices = network.getObservedVertices();
        checkUpstreamOfObservedDoesNotContainProbabilistic(observedVertices);

        List<Vertex> verticesToSampleFrom = variablesToSampleFrom.stream().map(v -> (Vertex) v).collect(Collectors.toList());

        Set<Vertex> allDownstreamVertices = allDownstreamVertices(network.getLatentVertices());
        Set<Vertex> transitiveClosureSampleFrom = TransitiveClosure.getUpstreamVerticesForCollection(verticesToSampleFrom, true).getAllVertices();
        Set<Vertex> intersection = Sets.intersection(allDownstreamVertices, transitiveClosureSampleFrom);

        List<Vertex> sortedVertices = TopologicalSort.sort(intersection);

        return new ForwardSampler(verticesToSampleFrom, sortedVertices, random);
    }

    private Set<Vertex> allDownstreamVertices(List<Vertex> randomVertices) {
        return LambdaSection.getDownstreamLambdaSectionForCollection(randomVertices, true).getAllVertices();
    }

    private void checkUpstreamOfObservedDoesNotContainProbabilistic(List<Vertex> observedVertices) {
        LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSectionForCollection(observedVertices, false);
        Set<Vertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();
        Preconditions.checkArgument(upstreamRandomVariables.size() == 1 || upstreamRandomVariables.size() == 0, "Forward sampler cannot be ran if observed variables have a random variable in their upstream lambda section");
    }
}
