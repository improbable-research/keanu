package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.KeanuProbabilisticModel;

import java.util.function.Function;

public class SamplingModelFitting {
    private final Function<KeanuProbabilisticModel, PosteriorSamplingAlgorithm> samplingAlgorithmGenerator;
    private final int samplingCount;
    private SamplingModelFitter fitter;

    public SamplingModelFitting(Function<KeanuProbabilisticModel, PosteriorSamplingAlgorithm> samplingAlgorithmGenerator, int samplingCount) {
        this.samplingAlgorithmGenerator = samplingAlgorithmGenerator;
        this.samplingCount = samplingCount;
    }

    public ModelFitter createFitterForGraph() {
        fitter = new SamplingModelFitter(samplingAlgorithmGenerator, samplingCount);
        return fitter;
    }

    public NetworkSamples getNetworkSamples() {
        return fitter.getNetworkSamples();
    }
}
