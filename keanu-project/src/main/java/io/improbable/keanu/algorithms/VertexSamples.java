package io.improbable.keanu.algorithms;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

public class VertexSamples<T> {

    protected final List<T> samples;

    public VertexSamples(List<T> samples) {
        this.samples = samples;
    }

    public double probability(Function<T, Boolean> samplePredicate) {
        long trueCount = samples.parallelStream().filter(samplePredicate::apply).count();

        return (double) trueCount / samples.size();
    }

    public T getMode() {

        if (samples.isEmpty()) {
            throw new IllegalStateException("Mode for empty samples is undefined");
        }

        Map<T, List<T>> groupedByValue = samples.stream().collect(groupingBy(v -> v));

        Optional<T> mode =
                groupedByValue
                        .entrySet()
                        .stream()
                        .sorted(comparing(v -> -v.getValue().size()))
                        .map(Map.Entry::getKey)
                        .findFirst();

        if (mode.isPresent()) {
            return mode.get();
        } else {
            throw new IllegalStateException("Mode is undefined");
        }
    }

    public List<T> asList() {
        return new ArrayList<>(samples);
    }
}
