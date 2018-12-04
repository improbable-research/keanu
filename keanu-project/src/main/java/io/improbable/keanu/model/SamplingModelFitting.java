package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class SamplingModelFitting<OUTPUT> {
    private final PosteriorSamplingAlgorithm samplingAlgorithm;
    private final int samplingCount;
    private SamplingModelFitter<DoubleTensor, OUTPUT> fitter;

    public SamplingModelFitting(PosteriorSamplingAlgorithm samplingAlgorithm, int samplingCount) {
        this.samplingAlgorithm = samplingAlgorithm;
        this.samplingCount = samplingCount;
    }

    public ModelFitter createFitterForGraph() {
        fitter = new SamplingModelFitter(samplingAlgorithm, samplingCount);
        return fitter;
    }

    public NetworkSamples getNetworkSamples() {
        return fitter.getNetworkSamples();
    }
}
