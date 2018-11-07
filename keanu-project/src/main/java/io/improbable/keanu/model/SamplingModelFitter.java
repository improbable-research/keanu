package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

public class SamplingModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;
    private final PosteriorSamplingAlgorithm samplingAlgorithm;
    private final int sampleCount;
    private final int dropCount;

    public SamplingModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph, PosteriorSamplingAlgorithm samplingAlgorithm, int sampleCount, int dropCount) {
        this.modelGraph = modelGraph;
        this.samplingAlgorithm = samplingAlgorithm;
        this.sampleCount = sampleCount;
        this.dropCount = dropCount;
    }

    /**
     * Uses the Metropolis Hastings sampling algorithm to fit the model graph to a given set of input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance, a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}
     *
     * @param input  The input data to your model graph
     * @param output The output data to your model graph
     */
    @Override
    public void fit(INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
        NetworkSamples posteriorSamples = samplingAlgorithm
            .getPosteriorSamples(modelGraph.getBayesianNetwork(), sampleCount)
            .drop(dropCount)
            .downSample(2);
        for (Vertex<DoubleTensor> vertex : modelGraph.getBayesianNetwork().getTopLevelLatentVertices()) {
            vertex.setValue(posteriorSamples.getDoubleTensorSamples(vertex).getAverages());
        }
    }
}
