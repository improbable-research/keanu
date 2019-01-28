package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase.NonGradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class SumGaussianTestCase implements GradientOptimizationAlgorithmTestCase, NonGradientOptimizationAlgorithmTestCase {

    private final DoubleVertex A;
    private final DoubleVertex B;

    private boolean useMLE;
    private KeanuProbabilisticWithGradientGraph model;

    public SumGaussianTestCase(boolean useMLE) {
        this.useMLE = useMLE;

        A = new GaussianVertex(20.0, 1.0);
        B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);
    }

    private void assertMLE(OptimizedResult result) {

        double maxA = result.get(A.getReference()).scalar();
        double maxB = result.get(B.getReference()).scalar();

        assertEquals(46, maxA + maxB, 0.1);
    }

    private void assertMAP(OptimizedResult result) {
        double maxA = result.get(A.getReference()).scalar();
        double maxB = result.get(B.getReference()).scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }

    @Override
    public FitnessFunction getFitnessFunction() {
        if (useMLE) {
            return new LogLikelihoodFitnessFunction(model);
        } else {
            return new LogProbFitnessFunction(model);
        }
    }

    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {
        return new FitnessFunctionGradient(model, useMLE, (a, b) -> {
        });
    }

    @Override
    public List<? extends Variable> getVariables() {
        return model.getLatentVariables();
    }

    @Override
    public void assertResult(OptimizedResult result) {
        if (useMLE) {
            assertMLE(result);
        } else {
            assertMAP(result);
        }
    }
}
