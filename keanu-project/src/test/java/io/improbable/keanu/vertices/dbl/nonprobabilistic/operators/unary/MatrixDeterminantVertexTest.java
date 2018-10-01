package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class MatrixDeterminantVertexTest {
    @Test
    public void calculatesDeterminant() {
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));
        Assert.assertThat(input.matrixDeterminant().getValue(), TensorMatchers.isScalarWithValue(-2d));
    }

    @Test
    public void canDifferentiateWrtScalar() {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final DoubleVertex output = input.matrixDeterminant().plus(1);
        final DoubleTensor expectedDerivative = DoubleTensor.create(new double[]{4, -3, -2, 1}, shape);
        assertReverseAutoDiffMatches(output, input, expectedDerivative);
    }

    @Test
    public void canDifferentiateWrtTensor() {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final DoubleVertex output = input.matrixDeterminant().times(input);

        // For calculation of derivative, see
        // https://www.wolframalpha.com/input/?i=x+%3D+1;+y+%3D+2;+z+%3D+3;+w+%3D+4;+d%2Fdx+(%7B%7Bx,y%7D,%7Bz,w%7D%7D+*+det(%7B%7Bx,y%7D,%7Bz,w%7D%7D))
        final double[] expectedDerivativeValues = new double[]{2, -3, -2, 1, 8, -8, -4, 2, 12, -9, -8, 3, 16, -12, -8, 2};
        final DoubleTensor expectedDerivative = DoubleTensor.create(expectedDerivativeValues, 2, 2, 2, 2);
        assertReverseAutoDiffMatches(output, input, expectedDerivative);
    }

    @Test
    public void canOptimiseOutOfTheBox() {
        assertOptimizerWorks(2);
    }

    @Test
    public void canOptimiseOutOfTheBoxStartingAtZero() {
        assertOptimizerWorks(0);
    }

    private void assertReverseAutoDiffMatches(DoubleVertex output, DoubleVertex input, DoubleTensor expectedDerivative) {
        final DoubleTensor derivative = Differentiator.reverseModeAutoDiff(output, input).withRespectTo(input);
        Assert.assertThat(derivative, TensorMatchers.elementwiseEqualTo(expectedDerivative));
    }

    private void assertOptimizerWorks(double inputGaussianMu) {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new GaussianVertex(shape, inputGaussianMu, 5);
        final DoubleVertex determinant = input.matrixDeterminant();
        final DoubleVertex output = new GaussianVertex(determinant, 1);
        output.observe(new double[]{2.0, 2.4});
        final BayesianNetwork net = new BayesianNetwork(output.getConnectedGraph());

        Optimizer.of(net).maxLikelihood();
        Assert.assertEquals(input.getValue().determinant(), 2.2, 0.1);
    }
}
