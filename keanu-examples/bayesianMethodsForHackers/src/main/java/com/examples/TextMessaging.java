package com.examples;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

/**
 * When did the author's text messaging rate increase, based on daily messaging counts?
 * <p>
 * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb#Example:-Inferring-behaviour-from-text-message-data
 */
public class TextMessaging {

    public static TextMessagingResults run() {

        final TextMessagingData data = ReadCsv.fromResources("text_messaging_data.csv")
            .asVectorizedColumnsDefinedBy(TextMessagingData.class)
            .load();

        final int numberOfDays = (int) data.numberOfMessages.getLength();

        // These hyperparameters differ from the alpha used in the example book
        // This is because the sampling algorithm of choice uses the prior distribution
        // as its proposal distribution. The suggested parameters were too wide, resulting
        // in bad proposals and by extension bad samples.
        final double alpha = 10;

        ExponentialVertex earlyRate = new ExponentialVertex(alpha);
        ExponentialVertex lateRate = new ExponentialVertex(alpha);
        UniformIntVertex switchPoint = new UniformIntVertex(0, numberOfDays);

        IntegerVertex days = ConstantVertex.of(data.day);
        DoubleVertex rateForDay = If.isTrue(new GreaterThanVertex<>(switchPoint, days))
            .then(earlyRate)
            .orElse(lateRate);

        PoissonVertex textsForDay = new PoissonVertex(rateForDay);
        textsForDay.observe(data.numberOfMessages);

        BayesianNetwork net = new BayesianNetwork(textsForDay.getConnectedGraph());
        net.probeForNonZeroProbability(1000);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(net);

        final int numSamples = 1000;
        NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model)
            .generatePosteriorSamples(model, net.getLatentVertices())
            .dropCount(numSamples / 2)
            .downSampleInterval(net.getLatentVertices().size())
            .generate(numSamples);

        return new TextMessagingResults(
            posteriorSamples.getIntegerTensorSamples(switchPoint).getScalarMode(),
            posteriorSamples.getDoubleTensorSamples(earlyRate).getAverages().scalar(),
            posteriorSamples.getDoubleTensorSamples(lateRate).getMode().scalar()
        );
    }

    public static class TextMessagingResults {
        public final int switchPointMode;
        public final double earlyRateMean;
        public final double lateRateMean;

        TextMessagingResults(int switchPointMode, double earlyRateMean, double lateRateMean) {
            this.switchPointMode = switchPointMode;
            this.earlyRateMean = earlyRateMean;
            this.lateRateMean = lateRateMean;
        }
    }

    public static class TextMessagingData {
        public IntegerTensor day;
        public IntegerTensor numberOfMessages;
    }
}
