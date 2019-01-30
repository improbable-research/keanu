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
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class SingleGaussianTestCase implements GradientOptimizationAlgorithmTestCase, NonGradientOptimizationAlgorithmTestCase {

    private GaussianVertex A;
    private long[] shape;

    private boolean useMLE;
    private KeanuProbabilisticModelWithGradient model;

    public SingleGaussianTestCase() {
        this(false, new long[]{2});
    }

    public SingleGaussianTestCase(boolean useMLE, long[] shape) {
        this.useMLE = useMLE;
        this.shape = shape;

        A = new GaussianVertex(shape, 10, 0.1);
        A.setValue(DoubleTensor.zeros(shape));

        GaussianVertex B = new GaussianVertex(A, 0.1);
        B.observe(DoubleTensor.create(5, shape));

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());
        model = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
    }

    private void assertMLE(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(5.0, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
    }


    private void assertMAP(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(7.5, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
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
