package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.LogLikelihoodFitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.LogProbFitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.LogLikelihoodFitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.LogProbFitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase.NonGradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class SumGaussianTestCase implements GradientOptimizationAlgorithmTestCase, NonGradientOptimizationAlgorithmTestCase {

    private final DoubleVertex A;
    private final DoubleVertex B;

    private boolean useMLE;
    private KeanuProbabilisticModelWithGradient model;

    public SumGaussianTestCase(boolean useMLE) {
        this.useMLE = useMLE;

        A = new GaussianVertex(20.0, 1.0);
        B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        model = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
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
        if (useMLE) {
            return new LogLikelihoodFitnessFunctionGradient(model);
        } else {
            return new LogProbFitnessFunctionGradient(model);
        }
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
