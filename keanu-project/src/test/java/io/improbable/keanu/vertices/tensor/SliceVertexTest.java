package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertEquals;

public class SliceVertexTest {

    private DoubleVertex matrixA;

    @Before
    public void setup() {
        matrixA = new ConstantDoubleVertex(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }

    @Test
    public void canDoAutodiffAcrossSliceAndConcat() {
        GaussianVertex latent = new GaussianVertex(new long[]{4, 1}, 0.0, 1.0);
        DoubleVertex[] slices = new DoubleVertex[4];

        for (int i = 0; i < slices.length; i++) {
            slices[i] = latent.slice(0, i);
        }

        ConcatenationVertex concatenationVertex = new ConcatenationVertex(0, slices);
        GaussianVertex observed = new GaussianVertex(concatenationVertex, 1);

        DoubleTensor dForward = Differentiator.forwardModeAutoDiff(latent, concatenationVertex).of(concatenationVertex);
        DoubleTensor dReverse = Differentiator.reverseModeAutoDiff(concatenationVertex, latent).withRespectTo(latent);

        assertEquals(dForward, dReverse);
    }

    @Test
    public void canGetTensorAlongDimensionOfRank2() {
        DoubleVertex rowOne = matrixA.slice(0, 0);

        Assert.assertArrayEquals(new double[]{1, 2, 3}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        DoubleVertex rowTwo = matrixA.slice(0, 1);

        Assert.assertArrayEquals(new double[]{4, 5, 6}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        DoubleVertex columnOne = matrixA.slice(1, 0);

        Assert.assertArrayEquals(new double[]{1, 4}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        DoubleVertex columnTwo = matrixA.slice(1, 1);

        Assert.assertArrayEquals(new double[]{2, 5}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        DoubleVertex columnThree = matrixA.slice(1, 2);

        Assert.assertArrayEquals(new double[]{3, 6}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canRepeatablySliceForAPick() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex columnZero = m.slice(0, 0);
        DoubleVertex elementZero = columnZero.slice(0, 0);

        assertEquals(elementZero.getValue().scalar(), 1, 1e-6);
    }

    @Test
    public void sliceCorrectlySplitsRowOfPartialDerivativeDimZeroIndexZero() {
        assertPartialsOfSliceWithRespectToOriginalAreCorrect(0, 0, new double[]{140, 170}, new long[]{2}, new long[]{2, 2, 3});
    }

    @Test
    public void sliceCorrectlySplitsRowOfPartialDerivativeDimZeroIndexOne() {
        assertPartialsOfSliceWithRespectToOriginalAreCorrect(0, 1, new double[]{320, 395}, new long[]{2}, new long[]{2, 2, 3});
    }

    @Test
    public void sliceCorrectlySplitsRowOfPartialDerivativeDimOneIndexZero() {
        assertPartialsOfSliceWithRespectToOriginalAreCorrect(1, 0, new double[]{140, 320}, new long[]{2}, new long[]{2, 2, 3});
    }

    @Test
    public void sliceCorrectlySplitsRowOfPartialDerivativeDimOneIndexOne() {
        assertPartialsOfSliceWithRespectToOriginalAreCorrect(1, 1, new double[]{170, 395}, new long[]{2}, new long[]{2, 2, 3});
    }

    private void assertPartialsOfSliceWithRespectToOriginalAreCorrect(int dim, int ind, double[] expectedValue, long[] expectedShape, long[] expectedPartialShape) {
        UniformVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6}, 2, 3));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25, 30, 35}, 3, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        DoubleVertex sliceN = N.slice(dim, ind);

        DoubleTensor dSliceNWrtmForward = Differentiator.forwardModeAutoDiff(m, sliceN).of(sliceN);

        DoubleTensor dSliceNWrtmReverse = Differentiator.reverseModeAutoDiff(sliceN, ImmutableSet.of(m, alpha)).withRespectTo(m);

        DoubleTensor originalPartial = Differentiator.reverseModeAutoDiff(N, m).withRespectTo(m);

        Assert.assertArrayEquals(sliceN.getValue().asFlatDoubleArray(), expectedValue, 1e-6);
        Assert.assertArrayEquals(expectedShape, sliceN.getShape());

        Assert.assertArrayEquals(originalPartial.slice(dim, ind).asFlatDoubleArray(), dSliceNWrtmForward.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(expectedPartialShape, dSliceNWrtmForward.getShape());

        Assert.assertArrayEquals(originalPartial.slice(dim, ind).asFlatDoubleArray(), dSliceNWrtmReverse.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(expectedPartialShape, dSliceNWrtmReverse.getShape());
    }

    @Test
    public void sliceCorrectlySplitsColumnOfPartialDerivative() {
        DoubleVertex m = new UniformVertex(0, 10);
        m.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex alpha = new UniformVertex(0, 10);
        alpha.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex N = m.matrixMultiply(alpha);

        DoubleVertex sliceN = N.slice(1, 1);

        DoubleTensor originalPartial = Differentiator.reverseModeAutoDiff(N, m).withRespectTo(m);
        DoubleTensor slicePartial = Differentiator.reverseModeAutoDiff(sliceN, m).withRespectTo(m);

        Assert.assertArrayEquals(sliceN.getValue().asFlatDoubleArray(), new double[]{65, 145}, 1e-6);
        Assert.assertArrayEquals(new long[]{2}, sliceN.getShape());

        Assert.assertArrayEquals(originalPartial.slice(1, 1).asFlatDoubleArray(), slicePartial.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2, 2}, slicePartial.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3() {
        DoubleVertex cube = new UniformVertex(0, 10);
        cube.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2}));

        DoubleVertex dimenZeroFace = cube.slice(0, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        DoubleVertex dimenOneFace = cube.slice(1, 0);
        Assert.assertArrayEquals(new double[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        DoubleVertex dimenTwoFace = cube.slice(2, 0);
        Assert.assertArrayEquals(new double[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex cube = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        DoubleVertex slice = cube.slice(2, 0);
        DoubleVertex result = slice.times(
            new ConstantDoubleVertex(new double[]{1., 2., 3., 4., 5., 6., 7., 8.}, new long[]{2, 2, 2})
        );
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(cube), result, 10.0, 1e-10);
    }

    @Test
    public void canGetTensorAlongDimensionOfRank2Integer() {
        IntegerVertex matrixA = new ConstantIntegerVertex(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6}, 2, 3));

        IntegerVertex rowOne = matrixA.slice(0, 0);

        Assert.assertArrayEquals(new int[]{1, 2, 3}, rowOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        IntegerVertex rowTwo = matrixA.slice(0, 1);

        Assert.assertArrayEquals(new int[]{4, 5, 6}, rowTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        IntegerVertex columnOne = matrixA.slice(1, 0);

        Assert.assertArrayEquals(new int[]{1, 4}, columnOne.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        IntegerVertex columnTwo = matrixA.slice(1, 1);

        Assert.assertArrayEquals(new int[]{2, 5}, columnTwo.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        IntegerVertex columnThree = matrixA.slice(1, 2);

        Assert.assertArrayEquals(new int[]{3, 6}, columnThree.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3Integer() {

        IntegerVertex cube = new ConstantIntegerVertex(0);
        cube.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        IntegerVertex dimenZeroFace = cube.slice(0, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, dimenZeroFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        IntegerVertex dimenOneFace = cube.slice(1, 0);
        Assert.assertArrayEquals(new int[]{1, 2, 5, 6}, dimenOneFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        IntegerVertex dimenTwoFace = cube.slice(2, 0);
        Assert.assertArrayEquals(new int[]{1, 3, 5, 7}, dimenTwoFace.getValue().asFlatIntegerArray());
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank2Boolean() {
        BooleanVertex matrixA = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, false, false, true, true}, 2, 3));
        BooleanVertex rowOne = matrixA.slice(0, 0);

        Assert.assertArrayEquals(new double[]{1, 1, 0}, rowOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowOne.getShape());

        BooleanVertex rowTwo = matrixA.slice(0, 1);

        Assert.assertArrayEquals(new double[]{0, 1, 1}, rowTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{3}, rowTwo.getShape());

        BooleanVertex columnOne = matrixA.slice(1, 0);

        Assert.assertArrayEquals(new double[]{1, 0}, columnOne.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnOne.getShape());

        BooleanVertex columnTwo = matrixA.slice(1, 1);

        Assert.assertArrayEquals(new double[]{1, 1}, columnTwo.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnTwo.getShape());

        BooleanVertex columnThree = matrixA.slice(1, 2);

        Assert.assertArrayEquals(new double[]{0, 1}, columnThree.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2}, columnThree.getShape());
    }

    @Test
    public void canGetTensorAlongDimensionOfRank3Boolean() {
        BooleanVertex cube = new ConstantBooleanVertex(false);
        cube.setValue(BooleanTensor.create(new boolean[]{true, true, false, false, true, true, false, false}, 2, 2, 2));

        BooleanVertex dimenZeroFace = cube.slice(0, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 0, 0}, dimenZeroFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenZeroFace.getShape());

        BooleanVertex dimenOneFace = cube.slice(1, 0);
        Assert.assertArrayEquals(new double[]{1, 1, 1, 1}, dimenOneFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenOneFace.getShape());

        BooleanVertex dimenTwoFace = cube.slice(2, 0);
        Assert.assertArrayEquals(new double[]{1, 0, 1, 0}, dimenTwoFace.getValue().asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(new long[]{2, 2}, dimenTwoFace.getShape());
    }


}
