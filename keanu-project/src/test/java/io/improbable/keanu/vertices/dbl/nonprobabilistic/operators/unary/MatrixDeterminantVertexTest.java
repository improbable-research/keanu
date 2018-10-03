package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;

import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.hamcrest.Matchers;
import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableList;

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
    public void calculatesDeterminantOnLargeMatrix() {
        final double[] values = new double[]{
            1, 8, 4, 5, 6,
            1, 100, 232, 4, 15,
            -10, 3, 57, 68, 10,
            7, 8, 9, 20, 10,
            12, 32, 43, -2, 5};
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(values, 5, 5));
        Assert.assertThat(input.matrixDeterminant().getValue(), TensorMatchers.isScalarWithValue(Matchers.closeTo(6120880d, 1e-5)));
    }

    @Test
    public void calculatesDeterminantOnScalar() {
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.scalar(5));
        Assert.assertThat(input.matrixDeterminant().getValue(), TensorMatchers.isScalarWithValue(5d));
    }

    @Test
    public void canDifferentiateWhenOutputIsScalar() {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final DoubleVertex output = input.matrixDeterminant();
        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(input), output, 0.001, 1e-5);
    }

    @Test
    public void canDifferentiateWhenOutputIsTensor() {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final DoubleVertex output = input.matrixDeterminant().times(input);

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(input), output, 0.001, 1e-5);
    }

    @Test(expected = IllegalArgumentException.class)
    public void failsForNonMatrixInputs() {
        final int[] shape = new int[]{2, 2, 2};
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(1, shape));
        input.matrixDeterminant();
    }

    @Test(expected = IllegalArgumentException.class)
    public void failsForNonSquareMatrices() {
        final int[] shape = new int[]{2, 3};
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(1, shape));
        input.matrixDeterminant();
    }

    @Test(expected = SingularMatrixException.class)
    public void differentiationFailsWhenMatrixIsSingular() {
        final int[] shape = new int[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{0, 0, 0,0}, shape));
        final DoubleVertex output = input.matrixDeterminant();
        Differentiator.reverseModeAutoDiff(output, input);
    }

    @Test
    public void canOptimiseOutOfTheBox() {
        assertOptimizerWorksWithDeterminant(2);
    }

    @Test
    public void canOptimiseOutOfTheBoxStartingAtZero() {
        assertOptimizerWorksWithDeterminant(0);
    }

    private void assertOptimizerWorksWithDeterminant(double inputGaussianMu) {
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
