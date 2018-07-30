package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.groupingBy;

public class IntegerTensorVertexSamples extends VertexSamples<IntegerTensor> {

    public IntegerTensorVertexSamples(List<IntegerTensor> samples) {
        super(samples);
    }

    public DoubleTensor getAverages() {
        if (samples.isEmpty()) {
            throw new IllegalStateException("No samples exist for averaging.");
        }

        int[] shape = samples.iterator().next().getShape();

        return this.samples.stream()
            .reduce(IntegerTensor.zeros(shape), IntegerTensor::plusInPlace)
            .toDouble()
            .divInPlace(samples.size());
    }

    public Integer getScalarMode() {
        return getModeAtIndex(0, 0);
    }

    public Integer getModeAtIndex(int... index) {

        if (samples.isEmpty()) {
            throw new IllegalStateException("Mode for empty samples is undefined");
        }

        Map<Integer, List<Integer>> groupedByValue = samples.stream()
            .map(v -> v.getValue(index))
            .collect(groupingBy(v -> v));

        Optional<Integer> mode = groupedByValue.entrySet().stream()
            .sorted(comparing(v -> -v.getValue().size()))
            .map(Map.Entry::getKey)
            .findFirst();

        if (mode.isPresent()) {
            return mode.get();
        } else {
            throw new IllegalStateException("Mode is undefined");
        }
    }
}
