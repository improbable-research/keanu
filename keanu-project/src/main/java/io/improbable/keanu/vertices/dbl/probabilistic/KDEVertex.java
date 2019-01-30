package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Samplable;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class KDEVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble, Samplable<DoubleTensor> {

    private final double bandwidth;
    private DoubleTensor samples;
    private static final String BANDWIDTH_NAME = "bandwidth";
    private static final String SAMPLES_NAME = "samples";

    @ExportVertexToPythonBindings
    public KDEVertex(@LoadVertexParam(SAMPLES_NAME) DoubleTensor samples, @LoadVertexParam(BANDWIDTH_NAME) double bandwidth) {
        super(Tensor.SCALAR_SHAPE);
        if (samples.getLength() == 0) {
            throw new IllegalStateException("The provided tensor of samples is empty!");
        }
        this.samples = samples;
        this.bandwidth = bandwidth;
    }

    public KDEVertex(DoubleTensor samples) {
        this(samples, scottsBandwidth(samples));
    }

    public KDEVertex(List<Double> samples) {
        this(DoubleTensor.create(samples.stream()
            .mapToDouble(Double::doubleValue)
            .toArray(), new long[]{samples.size()}));
    }

    public KDEVertex(List<Double> samples, double bandwidth) {
        this(DoubleTensor.create(samples.stream()
            .mapToDouble(Double::doubleValue)
            .toArray()), bandwidth);
    }

    @SaveVertexParam(BANDWIDTH_NAME)
    public double getBandwidth() {
        return bandwidth;
    }

    @SaveVertexParam(SAMPLES_NAME)
    public DoubleTensor getInputSamples() {
        return samples;
    }

    private DoubleTensor getDiffs(DoubleTensor x) {
        DoubleTensor diffs = DoubleTensor.zeros(samples.getShape()[0], x.getShape()[0]);
        return diffs.plusInPlace(x.reshape(1, x.getShape()[0])).minusInPlace(samples.reshape(samples.getShape()[0], 1)).divInPlace(bandwidth);
    }

    public DoubleTensor pdf(DoubleTensor x) {
        DoubleTensor diffs = getDiffs(x);
        return gaussianKernel(diffs).sum(0).divInPlace(samples.getLength() * bandwidth);
    }

    @Override
    public double logProb(DoubleTensor x) {
        return pdf(x).log().sum();
    }

    @Override
    public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor value, Set<? extends Vertex> withRespectTo) {
        Map<Vertex, DoubleTensor> partialDerivatives = new HashMap<>();

        if (withRespectTo.contains(this)) {
            DoubleTensor dlnPdfs = dPdx(value).divInPlace(pdf(value));
            partialDerivatives.put(this, dlnPdfs);
        }

        return partialDerivatives;
    }

    private DoubleTensor dPdx(DoubleTensor x) {
        DoubleTensor diff = getDiffs(x);
        return gaussianKernel(diff).timesInPlace(diff).unaryMinusInPlace().sum(0)
            .divInPlace(bandwidth * bandwidth * samples.getLength());
    }

    private DoubleTensor gaussianKernel(DoubleTensor x) {
        DoubleTensor power = x.pow(2.).timesInPlace(-0.5).expInPlace();
        return power.timesInPlace(1. / Math.sqrt(2. * Math.PI));
    }

    private static double scottsBandwidth(DoubleTensor samples) {
        return 1.06 * samples.standardDeviation() * Math.pow(samples.getLength(), -1. / 5.);
    }

    public DoubleTensor sample(int nSamples, KeanuRandom random) {
        // get a random sample as the mean of a gaussian
        // then draw a sample from the gaussian around that mean with the bandwidth as the standard deviation
        DoubleTensor value = Uniform.withParameters(
            DoubleTensor.scalar(0),
            DoubleTensor.scalar(samples.getLength())
        ).sample(new long[]{nSamples}, random);

        DoubleTensor index = value.floorInPlace();
        double[] shuffledSamples = new double[nSamples];
        int j = 0;
        for (Double i : index.asFlatList()) {
            shuffledSamples[j] = samples.getValue(i.intValue());
            j++;
        }

        DoubleTensor sampleMus = DoubleTensor.create(shuffledSamples);
        return random.nextGaussian(new long[]{nSamples}).timesInPlace(bandwidth).plusInPlace(sampleMus);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return sample(1, random);
    }

    public void resample(int nSamples, KeanuRandom random) {
        samples = sample(nSamples, random);
    }

    public long[] getSampleShape() {
        return samples.getShape();
    }

}