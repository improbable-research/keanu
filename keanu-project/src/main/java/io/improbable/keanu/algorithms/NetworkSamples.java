package io.improbable.keanu.algorithms;

import static java.util.stream.Collectors.toMap;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.intgr.IntegerTensorVertexSamples;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * An immutable collection of network samples. A network sample is a collection of values from
 * vertices in a network at a given point in time.
 */
public class NetworkSamples {

    private final Map<VertexId, ? extends List> samplesByVertex;
    private final List<Double> logOfMasterPForEachSample;
    private final int size;

    public NetworkSamples(
            Map<VertexId, ? extends List> samplesByVertex,
            List<Double> logOfMasterPForEachSample,
            int size) {
        this.samplesByVertex = samplesByVertex;
        this.logOfMasterPForEachSample = logOfMasterPForEachSample;
        this.size = size;
    }

    public int size() {
        return this.size;
    }

    public <T> VertexSamples<T> get(Vertex<T> vertex) {
        return get(vertex.getId());
    }

    public <T> VertexSamples<T> get(VertexId vertexId) {
        return new VertexSamples<>((List<T>) samplesByVertex.get(vertexId));
    }

    public DoubleVertexSamples getDoubleTensorSamples(Vertex<DoubleTensor> vertex) {
        return getDoubleTensorSamples(vertex.getId());
    }

    public DoubleVertexSamples getDoubleTensorSamples(VertexId vertexId) {
        return new DoubleVertexSamples(samplesByVertex.get(vertexId));
    }

    public IntegerTensorVertexSamples getIntegerTensorSamples(Vertex<IntegerTensor> vertex) {
        return getIntegerTensorSamples(vertex.getId());
    }

    public IntegerTensorVertexSamples getIntegerTensorSamples(VertexId vertexId) {
        return new IntegerTensorVertexSamples(samplesByVertex.get(vertexId));
    }

    public NetworkSamples drop(int dropCount) {

        final Map<VertexId, List<?>> withSamplesDropped =
                samplesByVertex
                        .entrySet()
                        .parallelStream()
                        .collect(
                                toMap(
                                        Map.Entry::getKey,
                                        e -> e.getValue().subList(dropCount, size)));
        final List<Double> withLogProbsDropped = logOfMasterPForEachSample.subList(dropCount, size);

        return new NetworkSamples(withSamplesDropped, withLogProbsDropped, size - dropCount);
    }

    public NetworkSamples downSample(final int downSampleInterval) {

        final Map<VertexId, List<?>> withSamplesDownSampled =
                samplesByVertex
                        .entrySet()
                        .parallelStream()
                        .collect(
                                toMap(
                                        Map.Entry::getKey,
                                        e ->
                                                downSample(
                                                        (List<?>) e.getValue(),
                                                        downSampleInterval)));
        final List<Double> withLogProbsDownSampled =
                downSample(logOfMasterPForEachSample, downSampleInterval);

        return new NetworkSamples(
                withSamplesDownSampled, withLogProbsDownSampled, size / downSampleInterval);
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
        long trueCount = networkStates.parallelStream().filter(predicate::apply).count();

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
            states.add(new SamplesBackedNetworkState(samplesByVertex, i));
        }
        return states;
    }

    private static class SamplesBackedNetworkState implements NetworkState {

        private final Map<VertexId, ? extends List> samplesByVertex;
        private final int index;

        public SamplesBackedNetworkState(Map<VertexId, ? extends List> samplesByVertex, int index) {
            this.samplesByVertex = samplesByVertex;
            this.index = index;
        }

        @Override
        public <T> T get(Vertex<T> vertex) {
            return ((List<T>) samplesByVertex.get(vertex.getId())).get(index);
        }

        @Override
        public <T> T get(VertexId vertexId) {
            return ((List<T>) samplesByVertex.get(vertexId)).get(index);
        }

        @Override
        public Set<VertexId> getVertexIds() {
            return new HashSet<>(samplesByVertex.keySet());
        }
    }
}
