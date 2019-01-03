package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexMatchers;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.junit.Assert;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.isScalarWithValue;
import static io.improbable.keanu.vertices.VertexMatchers.hasValue;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;
import static org.hamcrest.Matchers.closeTo;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class MatrixDeterminantVertexTest {
    @Test
    public void calculatesDeterminant() {
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));
        Assert.assertThat(input.matrixDeterminant(), hasValue(isScalarWithValue(-2d)));
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
        assertThat(input.matrixDeterminant(), VertexMatchers.hasValue(isScalarWithValue(closeTo(6120880d, 1e-5))));
    }

    @Test
    public void calculatesDeterminantOnOneByOne() {
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.scalar(5).reshape(1, 1));
        assertThat(input.matrixDeterminant(), hasValue(isScalarWithValue(5d)));
    }

    @Test
    public void canDifferentiateWhenOutputIsScalar() {
        final long[] shape = new long[]{2, 2};
        final UniformVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final MatrixDeterminantVertex output = input.matrixDeterminant();
        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(input), output, 0.001, 1e-5);
    }

    @Test
    public void canDifferentiateWhenOutputIsTensor() {
        final long[] shape = new long[]{2, 2};
        final UniformVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, shape));
        final MultiplicationVertex output = input.matrixDeterminant().times(input);

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(input), output, 0.001, 1e-5);
    }

    @Test(expected = IllegalArgumentException.class)
    public void failsForNonMatrixInputs() {
        final long[] shape = new long[]{2, 2, 2};
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(1, shape));
        input.matrixDeterminant();
    }

    @Test(expected = IllegalArgumentException.class)
    public void failsForNonSquareMatrices() {
        final long[] shape = new long[]{2, 3};
        final DoubleVertex input = new ConstantDoubleVertex(DoubleTensor.create(1, shape));
        input.matrixDeterminant();
    }

    @Test(expected = SingularMatrixException.class)
    public void differentiationFailsWhenMatrixIsSingular() {
        final long[] shape = new long[]{2, 2};
        final DoubleVertex input = new UniformVertex(shape, 0, 10);
        input.setValue(DoubleTensor.create(new double[]{0, 0, 0, 0}, shape));
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
        final long[] shape = new long[]{2, 2};
        final DoubleVertex input = new GaussianVertex(shape, inputGaussianMu, 5);
        final DoubleVertex determinant = input.matrixDeterminant();
        final DoubleVertex output = new GaussianVertex(determinant, 1);
        output.observe(new double[]{2.0, 2.4});
        final BayesianNetwork net = new BayesianNetwork(output.getConnectedGraph());

        KeanuOptimizer.of(net).maxLikelihood();
        assertEquals(input.getValue().determinant(), 2.2, 0.1);
    }
}
