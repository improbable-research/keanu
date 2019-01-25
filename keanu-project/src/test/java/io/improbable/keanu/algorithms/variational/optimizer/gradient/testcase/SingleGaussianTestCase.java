package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizerTestCase;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import static org.junit.Assert.assertArrayEquals;

public class SingleGaussianTestCase implements OptimizerTestCase {

    private GaussianVertex A;
    private BayesianNetwork model;
    private long[] shape;

    public SingleGaussianTestCase() {
        this(new long[]{2});
    }

    public SingleGaussianTestCase(long[] shape) {
        this.shape = shape;
        A = new GaussianVertex(shape, 10,0.1);
        A.setValue(DoubleTensor.zeros(shape));

        GaussianVertex B = new GaussianVertex(A, 0.1);
        B.observe(DoubleTensor.create(5, shape));

        model = new BayesianNetwork(A.getConnectedGraph());
    }

    @Override
    public BayesianNetwork getModel() {
        return model;
    }

    @Override
    public void assertMLE(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(5.0, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
    }

    @Override
    public void assertMAP(OptimizedResult result) {
        double[] maxA = result.get(A.getReference()).asFlatDoubleArray();

        DoubleTensor expected = DoubleTensor.create(7.5, shape);

        assertArrayEquals(expected.asFlatDoubleArray(), maxA, 1e-2);
    }
}
