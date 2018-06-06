package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;

public class DoubleVertexSamples extends VertexSamples<DoubleTensor> {

    public DoubleVertexSamples(List<DoubleTensor> samples) {
        super(samples);
    }

    public DoubleTensor getAverages() {
        if (samples.isEmpty()) {
            throw new IllegalStateException("No samples exist for averaging.");
        }

        int[] shape = samples.iterator().next().getShape();

        return this.samples.stream()
            .reduce(DoubleTensor.zeros(shape), DoubleTensor::plusInPlace)
            .divInPlace(samples.size());
    }

}
