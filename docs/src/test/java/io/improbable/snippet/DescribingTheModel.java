package io.improbable.snippet;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;

public class DescribingTheModel {

    public void doubleExample() {
        //%%SNIPPET_START%% DescribeVertex
        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);
        DoubleVertex C = A.plus(B);
        //%%SNIPPET_END%% DescribeVertex

        //%%SNIPPET_START%% DescribePropagate
        A.setAndCascade(1.234);
        //%%SNIPPET_END%% DescribePropagate

        //%%SNIPPET_START%% DescribeLazy
        A.setValue(1.234);
        B.setValue(4.321);
        C.lazyEval();
        //%%SNIPPET_END%% DescribeLazy


        //%%SNIPPET_START%% DescribeCascade
        A.setValue(1.234);
        B.setValue(4.321);
        VertexValuePropagation.cascadeUpdate(A, B);
        //%%SNIPPET_END%% DescribeCascade
    }

    public void booleanExample() {
        //%%SNIPPET_START%% DescribeAnd
        BooleanVertex A = new BernoulliVertex(0.5);
        BooleanVertex B = new BernoulliVertex(0.5);
        BooleanVertex C = A.and(B);
        //%%SNIPPET_END%% DescribeAnd

        //%%SNIPPET_START%% DescribeObserve
        C.observe(true);
        //%%SNIPPET_END%% DescribeObserve

        //%%SNIPPET_START%% DescribeInfer
        A.observe(true);
        B.observe(true);

        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());
        ProbabilisticModel model = new KeanuProbabilisticModel(net);
        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            model,
            Arrays.asList(A, B),
            100000
        ).drop(10000).downSample(2);
        double probabilityOfA = posteriorSamples.get(A).probability(isTrue -> isTrue.scalar() == true);
        //probabilityOfA evaluates to 1.0
        //%%SNIPPET_END%% DescribeInfer

        //%%SNIPPET_START%% DescribeInferIncorrect
        //WRONG
        A.lazyEval();
        B.lazyEval();
        System.out.println(A.getValue().scalar());
        //%%SNIPPET_END%% DescribeInferIncorrect
        //%%SNIPPET_START%% ProbeInfer
        BayesianNetwork bayesianNetwork = new BayesianNetwork(C.getConnectedGraph());
        bayesianNetwork.probeForNonZeroProbability(10);
        //%%SNIPPET_END%% ProbeInfer
    }
}
