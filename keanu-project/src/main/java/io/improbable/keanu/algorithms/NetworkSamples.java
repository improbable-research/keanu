package io.improbable.keanu.algorithms;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.intgr.IntegerVertexSamples;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toMap;

/**
 * An immutable collection of network samples. A network sample is a collection
 * of values from vertices in a network at a given point in time.
 */
@Slf4j
public class NetworkSamples {

    private final Map<VariableReference, ? extends List> samplesByVertex;
    private final List<Double> logOfMasterPForEachSample;
    private final int size;

    public NetworkSamples(Map<VariableReference, ? extends List> samplesByVertex, List<Double> logOfMasterPForEachSample, int size) {
        this.samplesByVertex = samplesByVertex;
        this.logOfMasterPForEachSample = logOfMasterPForEachSample;
        this.size = size;
    }

    public static NetworkSamples from(List<NetworkSample> networkSamples) {
        Map<VariableReference, List<?>> samplesByVertex = new HashMap<>();
        List<Double> logOfMasterPForEachSample = new ArrayList<>();

        networkSamples.forEach(networkSample -> addSamplesForNetworkSample(networkSample, samplesByVertex));
        networkSamples.forEach(networkSample -> logOfMasterPForEachSample.add(networkSample.getLogOfMasterP()));
        return new NetworkSamples(samplesByVertex, logOfMasterPForEachSample, networkSamples.size());
    }

    private static void addSamplesForNetworkSample(NetworkSample networkSample, Map<VariableReference, List<?>> samplesByVertex) {
        for (VariableReference vertexId : networkSample.getVertexIds()) {
            addSampleForVertex(vertexId, networkSample.get(vertexId), samplesByVertex);
        }
    }

    private static <T> void addSampleForVertex(VariableReference vertexId, T value, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertexId, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    public int size() {
        return this.size;
    }

    public <T> VertexSamples<T> get(Variable<T> vertex) {
        return get(vertex.getReference());
    }

    public <T> VertexSamples<T> get(VariableReference vertexId) {
        return new VertexSamples<>((List<T>) samplesByVertex.get(vertexId));
    }

    public DoubleVertexSamples getDoubleTensorSamples(Variable<DoubleTensor> vertex) {
        return getDoubleTensorSamples(vertex.getReference());
    }

    public DoubleVertexSamples getDoubleTensorSamples(VariableReference vertexId) {
        return new DoubleVertexSamples(samplesByVertex.get(vertexId));
    }

    public IntegerVertexSamples getIntegerTensorSamples(Variable<IntegerTensor> vertex) {
        return getIntegerTensorSamples(vertex.getReference());
    }

    public IntegerVertexSamples getIntegerTensorSamples(VariableReference vertexId) {
        return new IntegerVertexSamples(samplesByVertex.get(vertexId));
    }

    public NetworkSamples drop(int dropCount) {
        Preconditions.checkArgument(dropCount >= 0, "Cannot drop %s samples. Drop count must be positive.", dropCount);
        if (dropCount == 0) {
            return this;
        }

        final Map<VariableReference, List<?>> withSamplesDropped = samplesByVertex.entrySet().parallelStream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> e.getValue().subList(dropCount, size))
            );
        final List<Double> withLogProbsDropped = logOfMasterPForEachSample.subList(dropCount, size);

        return new NetworkSamples(withSamplesDropped, withLogProbsDropped, size - dropCount);
    }

    public NetworkSamples downSample(final int downSampleInterval) {
        Preconditions.checkArgument(downSampleInterval > 0, "Down sample interval of %s is invalid. Sample interval must be positive.", downSampleInterval);

        final Map<VariableReference, List<?>> withSamplesDownSampled = samplesByVertex.entrySet().parallelStream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> downSample((List<?>) e.getValue(), downSampleInterval)
                )
            );
        final List<Double> withLogProbsDownSampled = downSample(logOfMasterPForEachSample, downSampleInterval);

        return new NetworkSamples(withSamplesDownSampled, withLogProbsDownSampled, size / downSampleInterval);
    }

    private static <T> List<T> downSample(final List<T> samples, final int downSampleInterval) {

        List<T> downSampled = new ArrayList<>();
        int i = 0;

        for (T sample : samples) {
            if (i % downSampleInterval == 0) {
                downSampled.add(sample);
            }
            i++;
        }

        return downSampled;
    }

    public double probability(Function<NetworkState, Boolean> predicate) {
        List<NetworkState> networkStates = toNetworkStates();
        long trueCount = networkStates.parallelStream()
            .filter(predicate::apply)
            .count();

        return (double) trueCount / networkStates.size();
    }

    public NetworkState getNetworkState(int sample) {
        return new SamplesBackedNetworkState(samplesByVertex, sample);
    }

    public double getLogOfMasterP(int sample) {
        return logOfMasterPForEachSample.get(sample);
    }

    public List<NetworkState> toNetworkStates() {
        List<NetworkState> states = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            states.add(getNetworkState(i));
        }
        return states;
    }

    public NetworkState getMostProbableState() {
        Integer sampleNumberWithMostProbableState = IntStream.range(0, logOfMasterPForEachSample.size())
            .boxed().max(Comparator.comparing(i -> logOfMasterPForEachSample.get(i).doubleValue()))
            .orElse(0);
        log.debug(String.format("Most probable state is %d: %.4f",
            sampleNumberWithMostProbableState,
            logOfMasterPForEachSample.get(sampleNumberWithMostProbableState)));
        return getNetworkState(sampleNumberWithMostProbableState);
    }

    private static class SamplesBackedNetworkState implements NetworkState {

        private final Map<VariableReference, ? extends List> samplesByVertex;
        private final int index;

        public SamplesBackedNetworkState(Map<VariableReference, ? extends List> samplesByVertex, int index) {
            this.samplesByVertex = samplesByVertex;
            this.index = index;
        }

        @Override
        public <T> T get(Variable<T> vertex) {
            return ((List<T>) samplesByVertex.get(vertex.getReference())).get(index);
        }

        @Override
        public <T> T get(VariableReference vertexId) {
            return ((List<T>) samplesByVertex.get(vertexId)).get(index);
        }

        @Override
        public Set<VariableReference> getVertexIds() {
            return new HashSet<>(samplesByVertex.keySet());
        }
    }
}
