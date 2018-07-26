package examples;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import java.util.Arrays;

/**
 * Is treatment A significantly more likely to drive purchases than treatment B?
 * <p>
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb#Example:-Bayesian-A/B-testing
 */
public class ABTesting {
    private static KeanuRandom RANDOM = KeanuRandom.getDefaultRandom();

    public static ABTestingMaximumAPosteriori run() {

        DoubleVertex probabilityA = new UniformVertex(0., 1.);
        DoubleVertex probabilityB = new UniformVertex(0., 1.);

        DoubleVertex delta = probabilityA.minus(probabilityB);

        Flip observationA = new Flip(probabilityA);
        Flip observationB = new Flip(probabilityB);

        // manufacture observations
        int nObsA = 1500;
        BooleanTensor observationsA = RANDOM.nextDouble(new int[]{1, nObsA}).lessThan(0.05);
        observationA.observe(observationsA);

        int nObsB = 750;
        BooleanTensor observationsB = RANDOM.nextDouble(new int[]{1, nObsB}).lessThan(0.04);
        observationB.observe(observationsB);

        //infer the most probable probabilities
        BayesianNetwork net = new BayesianNetwork(probabilityA.getConnectedGraph());
        NetworkSamples networkSamples = MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(net, Arrays.asList(probabilityA, probabilityB, delta), 20000)
            .drop(1000).downSample(net.getLatentVertices().size());

        DoubleVertexSamples pASamples = networkSamples.getDoubleTensorSamples(probabilityA);
        DoubleVertexSamples pBSamples = networkSamples.getDoubleTensorSamples(probabilityB);

        //most probable probabilities are the averages of the MH walk in this case
        double mapPA = pASamples.getAverages().scalar();
        double mapPB = pBSamples.getAverages().scalar();

        return new ABTestingMaximumAPosteriori(mapPA, mapPB);
    }

    public static class ABTestingMaximumAPosteriori {
        public double pA;
        public double pB;

        public ABTestingMaximumAPosteriori(double pA, double pB) {
            this.pA = pA;
            this.pB = pB;
        }
    }
}
