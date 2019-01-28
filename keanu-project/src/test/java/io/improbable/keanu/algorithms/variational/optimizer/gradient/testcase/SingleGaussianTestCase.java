package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase.NonGradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class SingleGaussianTestCase implements GradientOptimizationAlgorithmTestCase, NonGradientOptimizationAlgorithmTestCase {

    private GaussianVertex A;
    private long[] shape;

    private boolean useMLE;
    private KeanuProbabilisticWithGradientGraph model;

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
        model = new KeanuProbabilisticWithGradientGraph(bayesianNetwork);
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
        return new FitnessFunction(model, useMLE, (a, b) -> {
        });
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
