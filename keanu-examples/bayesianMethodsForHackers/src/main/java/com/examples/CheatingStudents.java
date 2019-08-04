package com.examples;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.tensor.If;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic.BinomialVertex;

import static java.util.Collections.singletonList;

public class CheatingStudents {


    public static double runWithBernoulli(int numberOfStudents, int numberOfYesAnswers) {

        int numberOfSamples = 2000;
        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        BooleanVertex studentCheated = new BernoulliVertex(new long[]{numberOfStudents}, probabilityOfCheating);
        BooleanVertex answerIsTrue = new BernoulliVertex(new long[]{numberOfStudents}, 0.5);
        BooleanVertex randomAnswer = new BernoulliVertex(new long[]{numberOfStudents}, 0.5);

        DoubleVertex answer = If.isTrue(answerIsTrue)
            .then(
                If.isTrue(studentCheated)
                    .then(1.0)
                    .orElse(0.0)
            ).orElse(
                If.isTrue(randomAnswer)
                    .then(1.0)
                    .orElse(0.0)
            );

        DoubleVertex answerTotal = new GaussianVertex(answer.sum(), 1);
        answerTotal.observe(numberOfYesAnswers);

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(answerTotal.getConnectedGraph());

        NetworkSamplesGenerator samplesGenerator = Keanu.Sampling.MetropolisHastings.withDefaultConfig()
            .generatePosteriorSamples(model, singletonList(probabilityOfCheating));

        NetworkSamples networkSamples = samplesGenerator
            .dropCount(numberOfSamples / 2)
            .downSampleInterval(model.getLatentVariables().size())
            .generate(numberOfSamples);

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        return approximateProbabilityOfCheating;
    }

    public static double runUsingBinomial(int numberOfStudents, int numberOfYesAnswers) {
        int numberOfSamples = 100;

        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        DoubleVertex pYesAnswer = probabilityOfCheating.times(0.5).plus(0.25);
        BinomialVertex answerTotal = new BinomialVertex(pYesAnswer, numberOfStudents);
        answerTotal.observe(numberOfYesAnswers);

        KeanuProbabilisticModel model = new KeanuProbabilisticModel(answerTotal.getConnectedGraph());

        NetworkSamplesGenerator samplesGenerator = Keanu.Sampling.MetropolisHastings.withDefaultConfig()
            .generatePosteriorSamples(model, singletonList(probabilityOfCheating));

        NetworkSamples networkSamples = samplesGenerator.dropCount(numberOfSamples / 10)
            .downSampleInterval(model.getLatentVariables().size())
            .generate(numberOfSamples);

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        return approximateProbabilityOfCheating;
    }

}
