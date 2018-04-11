package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.algorithms.VertexSamples;

import java.util.DoubleSummaryStatistics;
import java.util.List;

public class DoubleVertexSamples extends VertexSamples<Double> {

    public DoubleVertexSamples(List<Double> samples) {
        super(samples);
    }

    public DoubleSummaryStatistics getSummaryStatistics() {
        return this.samples.stream().mapToDouble(v -> v).summaryStatistics();
    }
}
