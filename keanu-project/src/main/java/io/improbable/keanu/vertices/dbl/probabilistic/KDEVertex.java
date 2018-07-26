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

    public KDEVertex(List<Double> samples) {
        this(DoubleTensor.create(samples.stream()
            .mapToDouble(Double::doubleValue)
            .toArray()));
    }

    public KDEVertex(DoubleTensor samples) {
        this.samples = samples;
        this.bandwidth = scottsBandwith();
    }

    public DoubleTensor pdf(double x) {
        DoubleTensor xTensor = DoubleTensor.create(x, samples.getShape());
        return pdf(xTensor);
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
        // do a for loop here
        List<Double> valueAsList = value.asFlatList();
        double[] dlnPdfs = new double[valueAsList.size()];
        for (int i = 0; i < valueAsList.size(); i++){
            dlnPdfs[i] = dlnPdf(DoubleTensor.create(valueAsList.get(i), new int[]{1})).dLogPdx.scalar();
        }
        partialDerivates.put(getId(), DoubleTensor.create(dlnPdfs, new int[]{dlnPdfs.length}));
        return partialDerivates;
    }

    public DoubleTensor dPdx(double x) {
        DoubleTensor diff = DoubleTensor.create(x, samples.getShape()).minus(samples);
        return dPdx(diff.divInPlace(bandwidth)).sum(0, 1);
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

    private double getMean() {
        return samples.sum() / samples.getLength();
    }

    private double getVariance() {
        double mean = getMean();
        return samples.minus(DoubleTensor.create(mean, samples.getShape())).powInPlace(2).sum() / samples.getLength();
    }

    private double getStandardDeviation() {
        return Math.sqrt(getVariance());
    }

    private double scottsBandwith() {
        return 1.06 * getStandardDeviation() * Math.pow(samples.getLength(), -1. / 5.);
    }

    public DoubleTensor sample(int nSamples, KeanuRandom random) {
        // get a random sample as the mean of a gaussian
        // then draw a sample from the gaussian around that mean with the bandwidth as the standard deviation
        DoubleTensor value = Uniform.sample(new int[]{nSamples}, DoubleTensor.create(0, new int[]{1}), DoubleTensor.create(samples.getLength(), new int[]{1}), random);
        DoubleTensor index = value.floor();
        double[] newSamples = new double[nSamples];
        int j = 0;
        for (Double i : index.asFlatList()) {
            newSamples[j] = Gaussian.sample(new int[]{1}, DoubleTensor.create(samples.getValue(i.intValue()), new int[]{1}), DoubleTensor.create(bandwidth, new int[]{1}), random).scalar();
            j++;
        }
        return DoubleTensor.create(newSamples);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        // get a random sample as the mean of a gaussian
        // then draw a sample from the gaussian around that mean with the bandwidth as the standard deviation
        //double value = Uniform.sample(new int[]{1}, DoubleTensor.create(0, new int[]{1}), DoubleTensor.create(samples.getLength(), new int[]{1}), random).scalar();
        //int index = ((Double) Math.floor(value)).intValue();
        //return Gaussian.sample(new int[]{1}, DoubleTensor.create(samples.getValue(index), new int[]{1}), DoubleTensor.create(bandwidth, new int[]{1}), random);
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