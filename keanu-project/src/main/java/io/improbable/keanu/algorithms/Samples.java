package io.improbable.keanu.algorithms;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;

public class Samples<DATA, TENSOR extends Tensor<DATA>> {

    protected final List<TENSOR> samples;

    public Samples(List<TENSOR> samples) {
        Preconditions.checkArgument(!samples.isEmpty(), "No samples provided.");
        this.samples = samples;
    }

    public double probability(Function<TENSOR, Boolean> samplePredicate) {
        long trueCount = samples.parallelStream()
            .filter(samplePredicate::apply)
            .count();

        return (double) trueCount / samples.size();
    }

    public TENSOR getMode() {

        if (samples.isEmpty()) {
            throw new IllegalStateException("Mode for empty samples is undefined");
        }

        Map<TENSOR, List<TENSOR>> groupedByValue = samples.stream()
            .collect(groupingBy(v -> v));

        Optional<TENSOR> mode = groupedByValue.entrySet().stream()
            .sorted(comparing(v -> -v.getValue().size()))
            .map(Map.Entry::getKey)
            .findFirst();

        if (mode.isPresent()) {
            return mode.get();
        } else {
            throw new IllegalStateException("Mode is undefined");
        }
    }

    public List<TENSOR> asList() {
        return new ArrayList<>(samples);
    }

    public TENSOR asTensor() {
        List<DATA> data = new ArrayList<>();
        for (TENSOR sample : samples) {
            data.addAll(sample.asFlatList());
        }

        long[] sampleShape = samples.get(0).getShape();
        long[] shape = new long[1 + sampleShape.length];
        shape[0] = samples.size();
        System.arraycopy(sampleShape, 0, shape, 1, sampleShape.length);

        return (TENSOR) Tensor.create(data.toArray(), shape);
    }
}
