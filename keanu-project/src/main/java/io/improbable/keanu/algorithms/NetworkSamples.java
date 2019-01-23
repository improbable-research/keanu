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
 * of values from variables in a network at a given point in time.
 */
@Slf4j
public class NetworkSamples {

    private final Map<VariableReference, ? extends List> samplesByVariable;
    private final List<Double> logOfMasterPForEachSample;
    private final int size;

    public NetworkSamples(Map<VariableReference, ? extends List> samplesByVariable, List<Double> logOfMasterPForEachSample, int size) {
        this.samplesByVariable = samplesByVariable;
        this.logOfMasterPForEachSample = logOfMasterPForEachSample;
        this.size = size;
    }

    public static NetworkSamples from(List<NetworkSample> networkSamples) {
        Map<VariableReference, List<?>> samplesByVariable = new HashMap<>();
        List<Double> logOfMasterPForEachSample = new ArrayList<>();

        networkSamples.forEach(networkSample -> addSamplesForNetworkSample(networkSample, samplesByVariable));
        networkSamples.forEach(networkSample -> logOfMasterPForEachSample.add(networkSample.getLogOfMasterP()));
        return new NetworkSamples(samplesByVariable, logOfMasterPForEachSample, networkSamples.size());
    }

    private static void addSamplesForNetworkSample(NetworkSample networkSample, Map<VariableReference, List<?>> samplesByVariable) {
        for (VariableReference variableReference : networkSample.getVariableReferences()) {
            addSampleForVariable(variableReference, networkSample.get(variableReference), samplesByVariable);
        }
    }

    private static <T> void addSampleForVariable(VariableReference variableReference, T value, Map<VariableReference, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(variableReference, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

    public int size() {
        return this.size;
    }

    public <T> Samples<T> get(Variable<T, ?> variable) {
        return get(variable.getReference());
    }

    public <T> Samples<T> get(VariableReference variableReference) {
        return new Samples<>((List<T>) samplesByVariable.get(variableReference));
    }

    public DoubleVertexSamples getDoubleTensorSamples(Variable<DoubleTensor, ?> variable) {
        return getDoubleTensorSamples(variable.getReference());
    }

    public DoubleVertexSamples getDoubleTensorSamples(VariableReference variableReference) {
        return new DoubleVertexSamples(samplesByVariable.get(variableReference));
    }

    public IntegerVertexSamples getIntegerTensorSamples(Variable<IntegerTensor, ?> variable) {
        return getIntegerTensorSamples(variable.getReference());
    }

    public IntegerVertexSamples getIntegerTensorSamples(VariableReference variableReference) {
        return new IntegerVertexSamples(samplesByVariable.get(variableReference));
    }

    public NetworkSamples drop(int dropCount) {
        Preconditions.checkArgument(dropCount >= 0, "Cannot drop %s samples. Drop count must be positive.", dropCount);
        if (dropCount == 0) {
            return this;
        }

        final Map<VariableReference, List<?>> withSamplesDropped = samplesByVariable.entrySet().parallelStream()
            .collect(toMap(
                Map.Entry::getKey,
                e -> e.getValue().subList(dropCount, size))
            );
        final List<Double> withLogProbsDropped = logOfMasterPForEachSample.subList(dropCount, size);

        return new NetworkSamples(withSamplesDropped, withLogProbsDropped, size - dropCount);
    }

    public NetworkSamples downSample(final int downSampleInterval) {
        Preconditions.checkArgument(downSampleInterval > 0, "Down sample interval of %s is invalid. Sample interval must be positive.", downSampleInterval);

        final Map<VariableReference, List<?>> withSamplesDownSampled = samplesByVariable.entrySet().parallelStream()
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
        return new SamplesBackedNetworkState(samplesByVariable, sample);
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

        private final Map<VariableReference, ? extends List> samplesByVariable;
        private final int index;

        public SamplesBackedNetworkState(Map<VariableReference, ? extends List> samplesByVariable, int index) {
            this.samplesByVariable = samplesByVariable;
            this.index = index;
        }

        @Override
        public <T> T get(Variable<T, ?> variable) {
            return ((List<T>) samplesByVariable.get(variable.getReference())).get(index);
        }

        @Override
        public <T> T get(VariableReference variableReference) {
            return ((List<T>) samplesByVariable.get(variableReference)).get(index);
        }

        @Override
        public Set<VariableReference> getVariableReferences() {
            return new HashSet<>(samplesByVariable.keySet());
        }
    }
}
