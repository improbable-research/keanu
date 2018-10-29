package com.examples;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;

import static java.util.Collections.singletonList;

public class CheatingStudents {


    public static double runWithBernoulli(int numberOfStudents, int numberOfYesAnswers) {

        int numberOfSamples = 10000;
        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        BoolVertex studentCheated = new BernoulliVertex(new long[]{1, numberOfStudents}, probabilityOfCheating);
        BoolVertex answerIsTrue = new BernoulliVertex(new long[]{1, numberOfStudents}, 0.5);
        BoolVertex randomAnswer = new BernoulliVertex(new long[]{1, numberOfStudents}, 0.5);

        DoubleVertex answer = If.isTrue(answerIsTrue)
            .then(
                If.isTrue(studentCheated)
                    .then(1)
                    .orElse(0)
            ).orElse(
                If.isTrue(randomAnswer)
                    .then(1)
                    .orElse(0)
            );

        DoubleVertex answerTotal = new GaussianVertex(answer.sum(), 1);
        answerTotal.observe(numberOfYesAnswers);

        BayesianNetwork network = new BayesianNetwork(answerTotal.getConnectedGraph());

        NetworkSamplesGenerator samplesGenerator = MetropolisHastings.withDefaultConfig()
            .generatePosteriorSamples(network, singletonList(probabilityOfCheating));

        NetworkSamples networkSamples = samplesGenerator
            .dropCount(numberOfSamples / 10)
            .downSampleInterval(network.getLatentVertices().size())
            .generate(numberOfSamples);

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        return approximateProbabilityOfCheating;
    }

    public static double runUsingBinomial(int numberOfStudents, int numberOfYesAnswers) {
        int numberOfSamples = 10000;

        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        DoubleVertex pYesAnswer = probabilityOfCheating.times(0.5).plus(0.25);
        BinomialVertex answerTotal = new BinomialVertex(pYesAnswer, numberOfStudents);
        answerTotal.observe(numberOfYesAnswers);

        BayesianNetwork network = new BayesianNetwork(answerTotal.getConnectedGraph());

        NetworkSamplesGenerator samplesGenerator = MetropolisHastings.withDefaultConfig()
            .generatePosteriorSamples(network, singletonList(probabilityOfCheating));

        NetworkSamples networkSamples = samplesGenerator.dropCount(numberOfSamples / 10)
            .downSampleInterval(network.getLatentVertices().size())
            .generate(numberOfSamples);

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        return approximateProbabilityOfCheating;
    }

}
