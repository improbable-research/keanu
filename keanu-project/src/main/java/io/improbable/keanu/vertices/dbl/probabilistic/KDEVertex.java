package io.improbable.keanu.vertices.dbl.probabilistic;


import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KDEVertex extends ProbabilisticDouble {
    public double bandwidth;
    private DoubleTensor samples;

    public KDEVertex(DoubleTensor samples, double bandwidth){
        if (samples.getLength()==0){
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
            .toArray()));
    }

    public KDEVertex(List<Double> samples, double bandwidth) {
        this(DoubleTensor.create(samples.stream()
            .mapToDouble(Double::doubleValue)
            .toArray()), bandwidth);
    }

    public DoubleTensor pdf(DoubleTensor x) {
        List<Double> xAsList = x.asFlatList();
        double[] pdfs = new double[xAsList.size()];
        for (int i = 0; i < xAsList.size(); i++){
            DoubleTensor diff = DoubleTensor.create(xAsList.get(i), new int[]{1}).minus(samples);
            DoubleTensor gaussian = gaussianKernel(diff.divInPlace(bandwidth)).sum(0, 1);
            pdfs[i] = gaussian.divInPlace(DoubleTensor.create(samples.getLength() * bandwidth, gaussian.getShape())).scalar();
        }
        return DoubleTensor.create(pdfs);
    }

    @Override
    public double logPdf(DoubleTensor x) {
        return pdf(x).log().sum();
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        Map<Long, DoubleTensor> partialDerivates = new HashMap<>();
        
        List<Double> valueAsList = value.asFlatList();
        double[] dlnPdfs = new double[valueAsList.size()];
        for (int i = 0; i < valueAsList.size(); i++){
            dlnPdfs[i] = dlnPdf(DoubleTensor.create(valueAsList.get(i), new int[]{1})).dLogPdx.scalar();
        }
        partialDerivates.put(getId(), DoubleTensor.create(dlnPdfs, new int[]{dlnPdfs.length}));
        return partialDerivates;
    }

    private DoubleTensor dPdx(DoubleTensor x) {
        DoubleTensor diff = x.minus(samples).divInPlace(bandwidth);
        return gaussianKernel(diff).times(diff).unaryMinusInPlace().sum(0, 1) .divInPlace(bandwidth).divInPlace(bandwidth * samples.getLength());
    }

    public DiffLogP dlnPdf(DoubleTensor value) {
        return new DiffLogP(dPdx(value).divInPlace(pdf(value)));
    }

    private DoubleTensor gaussianKernel(DoubleTensor x) {
        DoubleTensor exponent = x.pow(2.).times(-1. / 2.);
        DoubleTensor power = DoubleTensor.create(Math.E, exponent.getShape()).powInPlace(exponent);
        power.timesInPlace(1. / Math.sqrt(2. * Math.PI));
        return power;
    }

    private static double getMean(DoubleTensor samples) {
        return samples.sum() / samples.getLength();
    }

    private static double getVariance(DoubleTensor samples) {
        double mean = getMean(samples);
        return samples.minus(DoubleTensor.create(mean, samples.getShape())).powInPlace(2).sum() / samples.getLength();
    }

    private static double getStandardDeviation(DoubleTensor samples) {
        return Math.sqrt(getVariance(samples));
    }

    private static double scottsBandwidth(DoubleTensor samples) {
        return 1.06 * getStandardDeviation(samples) * Math.pow(samples.getLength(), -1. / 5.);
    }

    public DoubleTensor sample(int nSamples, KeanuRandom random) {
        // get a random sample as the mean of a gaussian
        // then draw a sample from the gaussian around that mean with the bandwidth as the standard deviation
        DoubleTensor value = Uniform.withParameters(DoubleTensor.create(0, new int[]{1}), DoubleTensor.create(samples.getLength(), new int[]{1})).sample(new int[]{nSamples}, random);
        DoubleTensor index = value.floor();
        double[] newSamples = new double[nSamples];
        int j = 0;
        for (Double i : index.asFlatList()) {
            newSamples[j] = Gaussian.withParameters(DoubleTensor.create(samples.getValue(i.intValue()), new int[]{1}), DoubleTensor.create(bandwidth, new int[]{1})).sample(new int[]{nSamples}, random).scalar();
            j++;
        }
        return DoubleTensor.create(newSamples);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return sample(1, random);
    }

    public void resample(int nSamples, KeanuRandom random) {
        samples = sample(nSamples, random);
    }

    public static class DiffLogP {
        public final DoubleTensor dLogPdx;

        public DiffLogP(DoubleTensor dLogPdx) {
            this.dLogPdx = dLogPdx;
        }
    }

    public int[] getSampleShape() {
        return samples.getShape();
    }
}