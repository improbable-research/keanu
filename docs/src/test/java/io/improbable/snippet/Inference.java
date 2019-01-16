package io.improbable.snippet;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class Inference {
    private static void inferenceExample1() {
        //%%SNIPPET_START%% InfNormalDeclare
        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);
        //%%SNIPPET_END%% InfNormalDeclare

        //%%SNIPPET_START%% InfNoisyObservation
        DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
        C.observe(43.0);
        //%%SNIPPET_END%% InfNoisyObservation

        //%%SNIPPET_START%% InfStartState
        // Set the values of A and B so the Metropolis Hastings
        // algorithm has a non-zero start state.
        // Alternatively you could infer a possible start state
        // using either a particle filter or something like
        // bayesNet.probeForNonZeroProbability(100);
        A.setValue(20.0);
        B.setValue(20.0);

        BayesianNetwork bayesNet = new BayesianNetwork(C.getConnectedGraph());
        ProbabilisticModel model = new KeanuProbabilisticModel(bayesNet);
        //%%SNIPPET_END%% InfStartState

        //%%SNIPPET_START%% InfMetropolisHastings
        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            model,
            bayesNet.getLatentVertices(),
            100000
        );
        //%%SNIPPET_END%% InfMetropolisHastings

        //%%SNIPPET_START%% InfAverage
        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar(); //21.0
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar(); //21.0

        double actual = averagePosteriorA + averagePosteriorB; //42.0
        //%%SNIPPET_END%% InfAverage
    }

    private static void nutsExample() {
        BayesianNetwork bayesNet = null;
        ProbabilisticModel model = null;
        KeanuRandom random = null;

        //%%SNIPPET_START%% InfNuts
        NetworkSamples posteriorSamples = NUTS.withDefaultConfig().getPosteriorSamples(
            model,
            model.getLatentVariables(),
            2000
        );
        //%%SNIPPET_END%% InfNuts
    }
}
