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
import io.improbable.keanu.network.TransitiveClosure;
import io.improbable.keanu.util.status.StatusBar;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Vertex;
import org.nd4j.base.Preconditions;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Forward implements PosteriorSamplingAlgorithm {

    private final KeanuRandom random;
    private final boolean calculateSampleProbability;

    public static ForwardBuilder builder() {
        return new ForwardBuilder();
    }

    public Forward(KeanuRandom random, boolean calculateSampleProbability) {
        this.random = random;
        this.calculateSampleProbability = calculateSampleProbability;
    }

    /**
     * Samples from the prior of a Probabilistic Model.
     * <p>
     * Samples are taken by sampling from the prior of the desired variables in topological order.
     *
     * @param model                 the model to sample from
     * @param variablesToSampleFrom the variables to sample from
     * @param sampleCount           the number of samples to take
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

        List<IVertex> verticesToSampleFrom = new ArrayList<>();
        for (Variable variable : variablesToSampleFrom) {
            Preconditions.checkArgument(variable instanceof IVertex, "The Forward Sampler only works for Variables of type Vertex. Received : " + variable);
            verticesToSampleFrom.add((IVertex) variable);
        }

        BayesianNetwork network = checkSampleFromVariablesComeFromConnectedGraph(variablesToSampleFrom);

        List<IVertex> observedVertices = network.getObservedVertices();
        checkUpstreamOfObservedDoesNotContainProbabilistic(observedVertices);

        Set<IVertex> allDownstreamVertices = allDownstreamVertices(network.getLatentVertices());
        Set<IVertex> transitiveClosureSampleFrom = TransitiveClosure.getUpstreamVerticesForCollection(verticesToSampleFrom, true).getAllVertices();
        Set<IVertex> intersection = Sets.intersection(allDownstreamVertices, transitiveClosureSampleFrom);

        List<IVertex> sortedVertices = TopologicalSort.sort(intersection);

        return new ForwardSampler(network, verticesToSampleFrom, sortedVertices, random, calculateSampleProbability);
    }

    private BayesianNetwork checkSampleFromVariablesComeFromConnectedGraph(List<? extends Variable> variablesToSampleFrom) {
        Variable variable = variablesToSampleFrom.get(0);
        Set<IVertex> connectedGraph = ((IVertex) variable).getConnectedGraph();

        for (Variable var : variablesToSampleFrom) {
            if (!connectedGraph.contains(var)) {
                throw new IllegalArgumentException("Sample from vertices must be part of the same connected graph.");
            }
        }
        return new BayesianNetwork(connectedGraph);
    }

    private Set<IVertex> allDownstreamVertices(List<IVertex> randomVertices) {
        return LambdaSection.getDownstreamLambdaSectionForCollection(randomVertices, true).getAllVertices();
    }

    private void checkUpstreamOfObservedDoesNotContainProbabilistic(List<IVertex> observedVertices) {
        LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSectionForCollection(observedVertices, false);
        Set<IVertex> upstreamRandomVariables = upstreamLambdaSection.getAllVertices();
        if (upstreamRandomVariables.size() > 1) {
            throw new IllegalArgumentException("Forward sampler cannot be ran if observed variables have a random variable in their upstream lambda section");
        }
    }

    public static class ForwardBuilder {
        private KeanuRandom random = KeanuRandom.getDefaultRandom();
        private boolean calculateSampleProbability = false;

        ForwardBuilder() {
        }

        public ForwardBuilder random(KeanuRandom random) {
            this.random = random;
            return this;
        }

        public ForwardBuilder calculateSampleProbability(boolean calculateSampleProbability) {
            this.calculateSampleProbability = calculateSampleProbability;
            return this;
        }

        public Forward build() {
            return new Forward(random, calculateSampleProbability);
        }

        public String toString() {
            return "ForwardBuilder(random=" + this.random + ", calculateSampleProbability=" + this.calculateSampleProbability + ")";
        }
    }
}
