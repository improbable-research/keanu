package com.example;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.BetaVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex;

import java.util.Arrays;

public class CustomProbabilisticExample {

    public static void main(String[] args) {
        new CustomProbabilisticExample().doesCompile();
    }

    public void doesCompile() {

        DoubleVertex hyperPriors = new CustomProbabilisticVertex();

        //Set the starting values for your a and b. This is required especially since we didn't
        //implment a sample method for our CustomProbabilisticVertex
        hyperPriors.setValue(DoubleTensor.create(0.5, 0.5));

        DoubleVertex a = hyperPriors.slice(0, 0);
        DoubleVertex b = hyperPriors.slice(0, 1);

        DoubleVertex trueRates = new BetaVertex(new long[]{10}, a, b);

        IntegerVertex trials = ConstantVertex.of(100, 100, 100, 100, 100, 100, 100, 100, 100, 100);
        IntegerTensor successes = IntegerTensor.create(40, 44, 47, 54, 63, 46, 44, 49, 58, 50);

        BinomialVertex observedValues = new BinomialVertex(trueRates, trials);
        observedValues.observe(successes);

        BayesianNetwork model = new BayesianNetwork(observedValues.getConnectedGraph());

        //Use the gaussian proposal function with a sigma of 0.5. You can change this number and it will
        //effect the MCMC walk.
        PosteriorSamplingAlgorithm sampler = Keanu.Sampling.MetropolisHastings.builder()
            .proposalDistribution(new GaussianProposalDistribution(Arrays.asList(hyperPriors, trueRates), DoubleTensor.scalar(0.5)))
            .build();

        NetworkSamples samples = sampler
            .generatePosteriorSamples(new KeanuProbabilisticModel(model), model.getLatentVertices())
            .dropCount(500000)
            .generate(1000000);

        //Do some post processing of the samples. I've taken the average here but that probably isn't what you want.
        DoubleTensor averageTrueRates = samples.get(trueRates).asTensor().sum(0).div(500000);

        //Or you could find a MAP estimate for true rate. This is probably closer to what you want but I'm not sure
        //how well using MCMC will work to find the MAP for true rates.
        DoubleTensor mapOfTrueRates = samples.getMostProbableState().get(trueRates);

        System.out.println(mapOfTrueRates);

        //In summary, what you *probably* want to do here is use NUTS to infer true rates. In order to do that you
        //would need to implement the dLogProb in CustomProbabilisticVertex and use a different sampler like:

        sampler = Keanu.Sampling.NUTS.builder().maxTreeHeight(5).build();

        //This will not work until dLogProb is implemented in CustomProbabilisticVertex:
        NetworkSamples nutsSamples = sampler
            .generatePosteriorSamples(new KeanuProbabilisticModelWithGradient(model), model.getLatentVertices())
            .dropCount(1000)
            .generate(5000);

        DoubleTensor mapOfTrueRatesFromNUTS = nutsSamples.getMostProbableState().get(trueRates);
        System.out.println(mapOfTrueRatesFromNUTS);
    }
}
