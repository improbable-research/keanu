package io.improbable.keanu.vertices.dbl;

import com.google.common.base.Preconditions;
import io.improbable.keanu.algorithms.SampleStats;
import io.improbable.keanu.algorithms.VertexSamples;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;

public class DoubleVertexSamples extends VertexSamples<DoubleTensor> {
    private final long sampleShape[];

    public DoubleVertexSamples(List<DoubleTensor> samples) {
        super(samples);
        sampleShape = samples.iterator().next().getShape();
    }

    public DoubleTensor getAverages() {
        Preconditions.checkState(!samples.isEmpty(), "No samples exist for averaging.");
        Preconditions.checkArgument(!samples.isEmpty(), "No samples provided.");

        return this.samples.stream()
            .reduce(DoubleTensor.zeros(sampleShape), DoubleTensor::plusInPlace)
            .divInPlace(samples.size());
    }

    public DoubleTensor getVariances() {
        Preconditions.checkState(!samples.isEmpty(), "No samples exist for averaging.");

        DoubleTensor sumOfSquares = this.samples.stream()
            .reduce(DoubleTensor.zeros(sampleShape), (l, r) -> l.plusInPlace(r.pow(2)));
        return sumOfSquares
            .divInPlace(samples.size())
            .minusInPlace(getAverages().pow(2))
            .timesInPlace(samples.size())
            .divInPlace(samples.size() - 1);
    }

    /**
     * Calculates the autocorrelation of samples across a specified tensor index.
     *
     * @param index The tensor index to calculate autocorrelation across.
     * @return A tensor of autocorrelation at different lags.
     */
    public DoubleTensor getAutocorrelation(long... index) {
        Preconditions.checkState(!samples.isEmpty(), "No samples exist for averaging.");
        TensorShapeValidation.checkIndexIsValid(sampleShape, index);

        double[] sampleValuesAtIndex = samples.stream()
            .mapToDouble(x -> x.getValue(index))
            .toArray();
        double[] autocorr = SampleStats.autocorrelation(sampleValuesAtIndex);
        return DoubleTensor.create(autocorr);
    }

}
