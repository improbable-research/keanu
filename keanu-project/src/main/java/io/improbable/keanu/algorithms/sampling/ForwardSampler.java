package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.SamplingAlgorithm;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;
import java.util.Map;

import static io.improbable.keanu.algorithms.mcmc.SamplingUtil.takeSamples;

public class ForwardSampler implements SamplingAlgorithm {

    private static final double LOG_PROB_OF_PRIOR = 0.;

    private final List<? extends Variable> variablesToSampleFrom;
    private final List<Vertex> topologicallySortedVertices;
    private final KeanuRandom random;

    public ForwardSampler(List<? extends Variable> variablesToSampleFrom, List<Vertex> topologicallySortedVertices, KeanuRandom random) {
        this.variablesToSampleFrom = variablesToSampleFrom;
        this.topologicallySortedVertices = topologicallySortedVertices;
        this.random = random;
    }

    @Override
    public void step() {
        for (Vertex vertex : topologicallySortedVertices) {
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
        logOfMasterPForEachSample.add(LOG_PROB_OF_PRIOR);
    }

    @Override
    public NetworkSample sample() {
        step();
        return new NetworkSample(SamplingAlgorithm.takeSample(variablesToSampleFrom), LOG_PROB_OF_PRIOR);
    }
}
