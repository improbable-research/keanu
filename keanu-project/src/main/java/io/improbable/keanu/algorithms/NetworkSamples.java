package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.intgr.IntegerTensorVertexSamples;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

/**
 * An immutable collection of network samples. A network sample is a collection
 * of values from vertices in a network at a given point in time.
 */
public class NetworkSamples {

    private final Map<Long, ? extends List> samplesByVertex;
    private final int size;

    public NetworkSamples(Map<Long, ? extends List> samplesByVertex, int size) {
        this.samplesByVertex = samplesByVertex;
        this.size = size;
    }

    public int size() {
        return this.size;
    }

    public <T> VertexSamples<T> get(Vertex<T> vertex) {
        return get(vertex.getId());
    }

    public <T> VertexSamples<T> get(long vertexId) {
        return new VertexSamples<>((List<T>) samplesByVertex.get(vertexId));
    }

    public DoubleVertexSamples getDoubleTensorSamples(Vertex<DoubleTensor> vertex) {
        return getDoubleTensorSamples(vertex.getId());
    }

    public DoubleVertexSamples getDoubleTensorSamples(long vertexId) {
        return new DoubleVertexSamples(samplesByVertex.get(vertexId));
    }

    public IntegerTensorVertexSamples getIntegerTensorSamples(Vertex<IntegerTensor> vertex) {
        return getIntegerTensorSamples(vertex.getId());
    }

    public IntegerTensorVertexSamples getIntegerTensorSamples(long vertexId) {
        return new IntegerTensorVertexSamples(samplesByVertex.get(vertexId));
    }

    public NetworkSamples drop(int dropCount) {

        final Map<Long, List<?>> withSamplesDropped = samplesByVertex.entrySet().parallelStream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> e.getValue().subList(dropCount, size))
            );

        return new NetworkSamples(withSamplesDropped, size - dropCount);
    }

    public NetworkSamples downSample(final int downSampleInterval) {

        final Map<Long, List<?>> withSamplesDownSampled = samplesByVertex.entrySet().parallelStream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> downSample(e.getValue(), downSampleInterval)
                )
            );

        return new NetworkSamples(withSamplesDownSampled, size / downSampleInterval);
    }

    private static List<?> downSample(final List<?> samples, final int downSampleInterval) {

        List<Object> downSampled = new ArrayList<>();
        int i = 0;

        for (Object sample : samples) {
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

    public List<NetworkState> toNetworkStates() {
        List<NetworkState> states = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            states.add(new SamplesBackedNetworkState(samplesByVertex, i));
        }
        return states;
    }

    private static class SamplesBackedNetworkState implements NetworkState {

        private final Map<Long, ? extends List> samplesByVertex;
        private final int index;

        public SamplesBackedNetworkState(Map<Long, ? extends List> samplesByVertex, int index) {
            this.samplesByVertex = samplesByVertex;
            this.index = index;
        }

        @Override
        public <T> T get(Vertex<T> vertex) {
            return ((List<T>) samplesByVertex.get(vertex.getId())).get(index);
        }

        @Override
        public <T> T get(long vertexId) {
            return ((List<T>) samplesByVertex.get(vertexId)).get(index);
        }

        @Override
        public Set<Long> getVertexIds() {
            return new HashSet<>(samplesByVertex.keySet());
        }
    }

}
