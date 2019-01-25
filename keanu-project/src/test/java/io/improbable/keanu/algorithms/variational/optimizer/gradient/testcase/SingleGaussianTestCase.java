package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizerTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import static org.junit.Assert.assertArrayEquals;

public class SingleGaussianTestCase implements OptimizerTestCase {

    private GaussianVertex A;

    @Override
    public BayesianNetwork getModel() {
        A = new GaussianVertex(new long[]{2}, 10, 1);
        A.setValue(DoubleTensor.zeros(2));

        GaussianVertex B = new GaussianVertex(A, 1);
        B.observe(DoubleTensor.create(5, new long[]{2}));

        return new BayesianNetwork(A.getConnectedGraph());
    }

    @Override
    public void assertMLE(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        assertArrayEquals(new double[]{5.0, 5.0}, maxA, 1e-2);
    }

    @Override
    public void assertMAP(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        assertArrayEquals(new double[]{7.5, 7.5}, maxA, 1e-2);
    }
}
