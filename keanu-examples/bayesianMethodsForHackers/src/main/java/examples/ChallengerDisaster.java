package examples;

import com.google.common.primitives.Booleans;
import data.ChallengerData;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Does temperature correlate with defects?
 *
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter2_MorePyMC/Ch2_MorePyMC_PyMC3.ipynb#Example:-Challenger-Space-Shuttle-Disaster-
 */
public class ChallengerDisaster {
    public static ChallengerPosteriors run() {
        GaussianVertex beta = new GaussianVertex(0, 0.001);
        GaussianVertex alpha = new GaussianVertex(0, 0.001);

        double[] temps = new double[ChallengerData.LAUNCH_EVENTS.length];
        IntStream.range(0, temps.length).forEach(i -> temps[i] = ChallengerData.LAUNCH_EVENTS[i][0]);
        boolean[] defects = new boolean[ChallengerData.LAUNCH_EVENTS.length];
        IntStream.range(0, defects.length).forEach(i -> defects[i] = ChallengerData.LAUNCH_EVENTS[i][1] == 1);

        DoubleVertex temp = new ConstantDoubleVertex(temps);
        DoubleVertex logisticOutput = createLogisticFunction(beta, alpha, temp);
        Flip defect = new Flip(logisticOutput);
        defect.observe(defects);

        BayesianNetwork net = new BayesianNetwork(defect.getConnectedGraph());

        NetworkSamples networkSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), 120000)
                .drop(100000).downSample(2);

        ChallengerPosteriors cp = new ChallengerPosteriors();
        cp.networkSamples = networkSamples;
        cp.vertices = net.getLatentVertices();
        cp.alphaSamples = networkSamples.get(alpha).asList().stream().map(s -> s.scalar()).collect(Collectors.toList());
        cp.betaSamples = networkSamples.get(beta).asList().stream().map(s -> s.scalar()).collect(Collectors.toList());
        cp.alphaMode = networkSamples.get(alpha).getMode().scalar();
        cp.betaMode = networkSamples.get(beta).getMode().scalar();
        cp.temperature = Arrays.stream(temps).boxed().collect(Collectors.toList());
        cp.defects = Booleans.asList(defects).stream().map(b -> b ? 1. : 0.).collect(Collectors.toList());
        return cp;
    }

    private static DoubleVertex createLogisticFunction(GaussianVertex beta, GaussianVertex alpha, DoubleVertex temp) {
        ConstantDoubleVertex one = new ConstantDoubleVertex(1.0);
        ConstantDoubleVertex e = new ConstantDoubleVertex(Math.E);
        return one.divideBy(one.plus(e.pow(beta.times(temp).plus(alpha))));
    }

    public static class ChallengerPosteriors {
        private NetworkSamples networkSamples;
        private List<Vertex> vertices;
        private List<Double> alphaSamples;
        private List<Double> betaSamples;
        private List<Double> temperature;
        private List<Double> defects;
        private Double alphaMode;
        private Double betaMode;

        public NetworkSamples getNetworkSamples() {
            return networkSamples;
        }

        public List<Vertex> getVertices() {
            return vertices;
        }

        public List<Double> getAlphaSamples() {
            return alphaSamples;
        }

        public List<Double> getBetaSamples() {
            return betaSamples;
        }

        public List<Double> getTemperature() {
            return temperature;
        }

        public List<Double> getDefects() {
            return defects;
        }

        public Double getAlphaMode() {
            return alphaMode;
        }

        public Double getBetaMode() {
            return betaMode;
        }
    }
}
