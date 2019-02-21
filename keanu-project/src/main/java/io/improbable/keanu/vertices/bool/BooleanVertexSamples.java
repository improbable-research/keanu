package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.algorithms.Samples;
import io.improbable.keanu.tensor.bool.BooleanTensor;

import java.util.List;

public class BooleanVertexSamples extends Samples<BooleanTensor> {

    public BooleanVertexSamples(List<BooleanTensor> samples) {
        super(samples);
    }

    public BooleanTensor asTensor() {
        return BooleanTensor.stack(0, samples.stream().toArray(BooleanTensor[]::new));
    }
}
