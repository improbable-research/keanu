package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.NetworkState;

import java.util.function.Function;

public class SamplingModelFitter implements ModelFitter {

    private final Function<KeanuProbabilisticModel, PosteriorSamplingAlgorithm> samplingAlgorithmGenerator;
    private final int sampleCount;
    private NetworkSamples posteriorSamples;

    /**
     * This fitter uses a {@link PosteriorSamplingAlgorithm}, in contrast to the {@link MAPModelFitter} and {@link MaximumLikelihoodModelFitter}, which use gradient methods.
     *
     * The model's latent vertices will have their values set to the average over the samples.
     *
     * @param samplingAlgorithmGenerator The algorithm to use, e.g. {@link io.improbable.keanu.algorithms.mcmc.MetropolisHastings}
     * @param sampleCount The number of sample points to take.
     */
    public SamplingModelFitter(Function<KeanuProbabilisticModel, PosteriorSamplingAlgorithm> samplingAlgorithmGenerator, int sampleCount) {
        this.samplingAlgorithmGenerator = samplingAlgorithmGenerator;
        this.sampleCount = sampleCount;
    }

    /**
     * Uses a posterior sampling algorithm (e.g. Metropolis Hastings) to fit the model graph to the input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance, a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}
     *
     */
    @Override
    public void fit(ModelGraph modelGraph) {
        KeanuProbabilisticModel probabilisticModel = new KeanuProbabilisticModel(modelGraph.getBayesianNetwork());
        posteriorSamples = samplingAlgorithmGenerator.apply(probabilisticModel)
            .getPosteriorSamples(probabilisticModel, sampleCount);
        NetworkState mostProbableState = posteriorSamples.getMostProbableState();
        modelGraph.getBayesianNetwork().setState(mostProbableState);
    }

    public NetworkSamples getNetworkSamples() {
        return posteriorSamples;
    }

}
