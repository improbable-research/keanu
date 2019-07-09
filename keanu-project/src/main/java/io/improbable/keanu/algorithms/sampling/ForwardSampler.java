package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.SamplingUtil.takeSamples;

public class ForwardSampler implements SamplingAlgorithm {

    //Set to zero as the Forward Sampler is not interested in the log prob of samples
    private static final double LOG_PROB_OF_PRIOR = 0.;

    private final BayesianNetwork network;
    private final List<? extends Variable> variablesToSampleFrom;
    private final List<IVertex> topologicallySortedVertices;
    private final KeanuRandom random;
    private final boolean calculateSampleProbability;

    public ForwardSampler(BayesianNetwork network, List<? extends Variable> variablesToSampleFrom, List<IVertex> topologicallySortedVertices, KeanuRandom random, boolean calculateSampleProbability) {
        this.network = network;
        this.variablesToSampleFrom = variablesToSampleFrom;
        this.topologicallySortedVertices = topologicallySortedVertices;
        this.random = random;
        this.calculateSampleProbability = calculateSampleProbability;
    }

    @Override
    public void step() {
        for (IVertex vertex : topologicallySortedVertices) {
            if (vertex instanceof Probabilistic) {
                vertex.setValue(((Probabilistic) vertex).sample(random));
            } else if (vertex instanceof NonProbabilistic) {
                vertex.setValue(((NonProbabilistic) vertex).calculate());
            } else {
                throw new IllegalArgumentException("Forward sampler can only operate on Probabilistic or NonProbabilistic vertices. Invalid Vertex: [" + vertex + "]");
            }
        }
    }

    @Override
    public void sample(Map<VariableReference, List<?>> samples, List<Double> logOfMasterPForEachSample) {
        step();
        takeSamples(samples, variablesToSampleFrom);
        double logProb = calculateSampleProbability ? network.getLogOfMasterP() : LOG_PROB_OF_PRIOR;
        logOfMasterPForEachSample.add(logProb);
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(SamplingAlgorithm.takeSample(variablesToSampleFrom), LOG_PROB_OF_PRIOR);
    }
}
