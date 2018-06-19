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

import java.util.List;
import java.util.stream.Collectors;

/**
 * Is treatment A significantly more likely to drive purchases than treatment B?
 *
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb#Example:-Bayesian-A/B-testing
 */
public class ABTesting {
    private static KeanuRandom RANDOM = KeanuRandom.getDefaultRandom();

    public static ABTestingPosteriors run() {
        DoubleVertex pA = new UniformVertex(0., 1.);
        DoubleVertex pB = new UniformVertex(0., 1.);

        DoubleVertex delta = pA.minus(pB);

        Flip obsA = new Flip(pA);
        Flip obsB = new Flip(pB);

        // manufacture observations
        int nObsA = 1500;
        BooleanTensor observationsA = RANDOM.nextDouble(new int[] {1, nObsA}).lessThan(0.05);
        obsA.observe(observationsA);
        double observedFrequencyA = (double)observationsA.asFlatList().stream().filter(b -> b).count() / nObsA;

        int nObsB = 750;
        BooleanTensor observationsB = RANDOM.nextDouble(new int[] {1, nObsB}).lessThan(0.04);
        obsB.observe(observationsB);
        double observedFrequencyB = (double)observationsB.asFlatList().stream().filter(b -> b).count() / nObsB;

        BayesianNetwork net = new BayesianNetwork(pA.getConnectedGraph());

        NetworkSamples networkSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), 20000)
                .drop(1000).downSample(1);

        DoubleVertexSamples pASamples = networkSamples.getDoubleTensorSamples(pA);
        DoubleVertexSamples pBSamples = networkSamples.getDoubleTensorSamples(pB);
        ABTestingPosteriors out = new ABTestingPosteriors();
        out.pASamples = pASamples.asList().stream().map(dT -> dT.scalar()).collect(Collectors.toList());
        out.pBSamples = pBSamples.asList().stream().map(dT -> dT.scalar()).collect(Collectors.toList());
        out.pAMode = pASamples.getMode().scalar();
        out.pBMode = pBSamples.getMode().scalar();
        return out;
    }

    public static class ABTestingPosteriors {
        private List<Double> pASamples;
        private List<Double> pBSamples;
        private Double pAMode;
        private Double pBMode;

        public List<Double> getpASamples() {
            return pASamples;
        }

        public List<Double> getpBSamples() {
            return pBSamples;
        }

        public Double getpAMode() {
            return pAMode;
        }

        public Double getpBMode() {
            return pBMode;
        }
    }
}
