package com.examples;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * Does temperature correlate with defects?
 * <p>
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb#Example:-Challenger-Space-Shuttle-Disaster-
 */
public class ChallengerDisaster {
    public static ChallengerPosteriors run() {
        ChallengerData data = ReadCsv.fromResources("challenger_data.csv")
            .asVectorizedColumnsDefinedBy(ChallengerData.class)
            .load();

        // These hyperparameters differ from the alpha used in the example book
        // This is because the sampling algorithm of choice uses the prior distribution
        // as its proposal distribution. The suggested parameters were too wide, resulting
        // in bad proposals and by extension bad samples.
        // When it is easier to decouple the prior from the proposal distribution, we should revisit this
        final double betaSigma = convertTauToSigma(0.01);
        final double alphaSigma = convertTauToSigma(0.005);

        GaussianVertex alpha = new GaussianVertex(0, alphaSigma);
        GaussianVertex beta = new GaussianVertex(0, betaSigma);

        DoubleVertex temps = new ConstantDoubleVertex(data.temps);
        DoubleVertex logisticOutput = createLogisticFunction(beta, alpha, temps);

        BernoulliVertex defect = new BernoulliVertex(logisticOutput);
        defect.observe(data.oRingFailure);

        BayesianNetwork net = new BayesianNetwork(defect.getConnectedGraph());
        net.probeForNonZeroProbability(1000);

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

        final int sampleCount = 3000;
        NetworkSamples networkSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model)
            .generatePosteriorSamples(model, model.getLatentVariables())
            .dropCount(sampleCount / 2)
            .downSampleInterval(model.getLatentVariables().size())
            .generate(sampleCount);

        ChallengerPosteriors cp = new ChallengerPosteriors();
        cp.mapAlpha = networkSamples.getDoubleTensorSamples(alpha).getAverages().scalar();
        cp.mapBeta = networkSamples.getDoubleTensorSamples(beta).getAverages().scalar();

        return cp;
    }

    private static double convertTauToSigma(double tau) {
        return Math.sqrt(1.0 / tau);
    }

    private static DoubleVertex createLogisticFunction(GaussianVertex beta, GaussianVertex alpha, DoubleVertex temp) {
        return beta.times(temp).plus(alpha).unaryMinus().sigmoid();
    }

    public static class ChallengerPosteriors {
        public Double mapAlpha;
        public Double mapBeta;
    }

    public static class ChallengerData {
        public DoubleTensor temps;
        public BooleanTensor oRingFailure;
    }
}
