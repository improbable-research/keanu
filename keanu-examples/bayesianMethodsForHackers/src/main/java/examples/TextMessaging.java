package examples;

import data.TextMessagingData;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

import java.util.List;
import java.util.stream.Collectors;

/**
 * When did the author's text messaging rate increase, based on daily messaging counts?
 *
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb#Example:-Inferring-behaviour-from-text-message-data
 */
public class TextMessaging {

    public static TextMessagingResults run() {
        double avgTexts = TextMessagingData.DAYS.stream().mapToDouble(i -> i).average().orElseThrow(RuntimeException::new);
        double alpha = 1 / avgTexts;

        ExponentialVertex earlyRate = new ExponentialVertex(alpha, alpha);
        ExponentialVertex lateRate = new ExponentialVertex(alpha, alpha);
        UniformIntVertex switchPoint = new UniformIntVertex(0, TextMessagingData.DAYS.size() + 1);
        ConstantIntegerVertex days = ConstantVertex.of(TextMessagingData.DAYS.size());

        DoubleVertex rateForDay = If.isTrue(new GreaterThanVertex<>(switchPoint, days)).then(earlyRate).orElse(lateRate);

        PoissonVertex textsForDay = new PoissonVertex(rateForDay);
        textsForDay.observe(TextMessagingData.DAYS.stream().map(i -> (int) (double) i).mapToInt(i -> i).toArray());

        BayesianNetwork net = new BayesianNetwork(switchPoint.getConnectedGraph());

        int numSamples = 5000;

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), numSamples);

        NetworkSamples results = posteriorSamples.drop(1000).downSample(3);

        TextMessagingResults out = new TextMessagingResults();
        out.earlyRateSamples = results.getDoubleTensorSamples(earlyRate).asList().stream().map(t -> t.scalar()).collect(Collectors.toList());
        out.lateRateSamples = results.getDoubleTensorSamples(lateRate).asList().stream().map(t -> t.scalar()).collect(Collectors.toList());
        out.switchPointSamples = results.getIntegerTensorSamples(switchPoint).asList().stream().map(t -> t.scalar()).collect(Collectors.toList());
        out.earlyRateMode = results.get(earlyRate).getMode().scalar();
        out.lateRateMode = results.get(lateRate).getMode().scalar();
        out.switchPointMode = results.get(switchPoint).getMode().scalar();
        return out;
    }

    public static class TextMessagingResults {
        private List<Double> earlyRateSamples;
        private List<Double> lateRateSamples;
        private List<Integer> switchPointSamples;
        private Double earlyRateMode;
        private Double lateRateMode;
        private Integer switchPointMode;

        public List<Double> getEarlyRateSamples() {
            return earlyRateSamples;
        }

        public List<Double> getLateRateSamples() {
            return lateRateSamples;
        }

        public List<Integer> getSwitchPointSamples() {
            return switchPointSamples;
        }

        public Double getEarlyRateMode() {
            return earlyRateMode;
        }

        public Double getLateRateMode() {
            return lateRateMode;
        }

        public Integer getSwitchPointMode() {
            return switchPointMode;
        }
    }
}
