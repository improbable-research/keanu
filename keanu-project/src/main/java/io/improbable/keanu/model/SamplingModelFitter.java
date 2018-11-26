package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.NetworkState;

public class SamplingModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;
    private final PosteriorSamplingAlgorithm samplingAlgorithm;
    private final int sampleCount;
    private NetworkSamples posteriorSamples;

    /**
     * This fitter uses a {@link PosteriorSamplingAlgorithm}, in contrast to the {@link MAPModelFitter} and {@link MaximumLikelihoodModelFitter}, which use gradient methods.
     *
     * The model's latent vertices will have their values set to the average over the samples.
     *
     * @param modelGraph The graph to fit
     * @param samplingAlgorithm The algorithm to use, e.g. {@link io.improbable.keanu.algorithms.mcmc.MetropolisHastings}
     * @param sampleCount The number of sample points to take.
     */
    public SamplingModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph, PosteriorSamplingAlgorithm samplingAlgorithm, int sampleCount) {
        this.modelGraph = modelGraph;
        this.samplingAlgorithm = samplingAlgorithm;
        this.sampleCount = sampleCount;
    }

    /**
     * Uses a posterior sampling algorithm (e.g. Metropolis Hastings) to fit the model graph to a given set of input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance, a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}
     *
     * @param input  The input data to your model graph
     * @param output The output data to your model graph
     */
    @Override
    public void fit(INPUT input, OUTPUT output) {
        posteriorSamples = samplingAlgorithm
            .getPosteriorSamples(modelGraph.getBayesianNetwork(), sampleCount);
        NetworkState mostProbableState = posteriorSamples.getMostProbableState();
        modelGraph.getBayesianNetwork().setState(mostProbableState);
    }

    @Override
    public void observe(INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
    }

    public NetworkSamples getNetworkSamples() {
        return posteriorSamples;
    }
}
