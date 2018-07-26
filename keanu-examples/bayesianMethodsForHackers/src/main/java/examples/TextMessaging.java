package examples;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

/**
 * When did the author's text messaging rate increase, based on daily messaging counts?
 * <p>
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb#Example:-Inferring-behaviour-from-text-message-data
 */
public class TextMessaging {

    public static TextMessagingResults run() {

        TextMessagingData data = ReadCsv.fromResources("text_messaging_data.csv")
            .asVectorizedColumnsDefinedBy(TextMessagingData.class)
            .load();

        int numberOfDays = (int) data.numberOfMessages.getLength();
        double avgTexts = (double) data.numberOfMessages.sum() / numberOfDays;
        double alpha = 1 / avgTexts;

        ExponentialVertex earlyRate = new ExponentialVertex(alpha, alpha);
        ExponentialVertex lateRate = new ExponentialVertex(alpha, alpha);
        UniformIntVertex switchPoint = new UniformIntVertex(0, numberOfDays + 1);

        DoubleVertex rateForDay = If.isTrue(new GreaterThanVertex<>(switchPoint, ConstantVertex.of(numberOfDays)))
            .then(earlyRate)
            .orElse(lateRate);

        PoissonVertex textsForDay = new PoissonVertex(rateForDay);
        textsForDay.observe(data.numberOfMessages);

        BayesianNetwork net = new BayesianNetwork(switchPoint.getConnectedGraph());

        int numSamples = 5000;
        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(net, net.getLatentVertices(), numSamples)
            .drop(numSamples / 10)
            .downSample(net.getLatentVertices().size());

        int mostProbableSwitchPoint = posteriorSamples.get(switchPoint).getMode().scalar();
        return new TextMessagingResults(mostProbableSwitchPoint);
    }

    public static class TextMessagingResults {
        public int switchPointMode;
        TextMessagingResults(int switchPointMode) {
            this.switchPointMode = switchPointMode;
        }
    }

    public static class TextMessagingData {
        public IntegerTensor numberOfMessages;
    }
}
