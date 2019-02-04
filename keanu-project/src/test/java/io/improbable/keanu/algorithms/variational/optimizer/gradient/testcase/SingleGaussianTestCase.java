package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilityFitness;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase.NonGradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class SingleGaussianTestCase implements GradientOptimizationAlgorithmTestCase, NonGradientOptimizationAlgorithmTestCase {

    private final GaussianVertex A;
    private final long[] shape;

    private final ProbabilityFitness probabilityFitness;
    private final KeanuProbabilisticModelWithGradient model;

    public SingleGaussianTestCase() {
        this(ProbabilityFitness.MAP, new long[]{2});
    }

    public SingleGaussianTestCase(ProbabilityFitness probabilityFitness, long[] shape) {
        this.probabilityFitness = probabilityFitness;
        this.shape = shape;

        A = new GaussianVertex(shape, 10, 0.1);
        A.setValue(DoubleTensor.zeros(shape));

        GaussianVertex B = new GaussianVertex(A, 0.1);
        B.observe(DoubleTensor.create(5, shape));

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());
        model = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
    }

    private void assertMLE(OptimizedResult result) {
        double[] maxA = result.getValueFor(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(5.0, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
    }


    private void assertMAP(OptimizedResult result) {
        double[] maxA = result.getValueFor(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(7.5, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
    }

    @Override
    public FitnessFunction getFitnessFunction() {
        return probabilityFitness.getFitnessFunction(model);
    }

    @Override
    public FitnessFunctionGradient getFitnessFunctionGradient() {
        return probabilityFitness.getFitnessFunctionGradient(model);
    }

    @Override
    public List<? extends Variable> getVariables() {
        return model.getLatentVariables();
    }

    @Override
    public void assertResult(OptimizedResult result) {
        if (probabilityFitness.equals(ProbabilityFitness.MLE)) {
            assertMLE(result);
        } else {
            assertMAP(result);
        }
    }
}
