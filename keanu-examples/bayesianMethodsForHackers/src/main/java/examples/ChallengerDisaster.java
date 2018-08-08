package examples;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
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

        double tau = 0.001;
        double sigma = Math.sqrt(1.0 / tau);
        GaussianVertex beta = new GaussianVertex(0, sigma);
        GaussianVertex alpha = new GaussianVertex(0, sigma);

        DoubleVertex temps = new ConstantDoubleVertex(data.temps);
        DoubleVertex logisticOutput = createLogisticFunction(beta, alpha, temps);

        BernoulliVertex defect = new BernoulliVertex(logisticOutput);
        defect.observe(data.oRingFailure);

        BayesianNetwork net = new BayesianNetwork(defect.getConnectedGraph());
        net.probeForNonZeroProbability(10000);

        int sampleCount = 120000;
        NetworkSamples networkSamples = MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(net, net.getLatentVertices(), sampleCount)
            .drop(sampleCount / 10).downSample(net.getLatentVertices().size());

        ChallengerPosteriors cp = new ChallengerPosteriors();
        cp.mapAlpha = networkSamples.getDoubleTensorSamples(alpha).getAverages().scalar();
        cp.mapBeta = networkSamples.getDoubleTensorSamples(beta).getAverages().scalar();

        return cp;
    }

    private static DoubleVertex createLogisticFunction(GaussianVertex beta, GaussianVertex alpha, DoubleVertex temp) {
        ConstantDoubleVertex one = new ConstantDoubleVertex(1.0);
        ConstantDoubleVertex e = new ConstantDoubleVertex(Math.E);
        return one.divideBy(one.plus(e.pow(beta.times(temp).plus(alpha))));
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
