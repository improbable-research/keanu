package io.improbable.keanu.tensor.dbl;

import com.google.common.primitives.Ints;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.Slicer;
import io.improbable.keanu.tensor.TensorFactories;
import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorTestHelper;
import io.improbable.keanu.tensor.TensorValueException;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import io.improbable.keanu.tensor.validate.policy.TensorValidationPolicy;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorMatchers.hasValue;
import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class DoubleTensorTest {

    @Parameterized.Parameters(name = "{index}: Test with {1}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {new Nd4jDoubleTensorFactory(), "ND4J DoubleTensor"},
            {new JVMDoubleTensorFactory(), "JVM DoubleTensor"},
        });
    }

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private DoubleTensor matrixA;
    private DoubleTensor matrixB;
    private DoubleTensor scalarA;
    private DoubleTensor vectorA;
    private DoubleTensor rankThreeTensor;

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    public DoubleTensorTest(DoubleTensorFactory factory, String name) {

        TensorFactories.doubleTensorFactory = factory;

        matrixA = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        matrixB = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        scalarA = DoubleTensor.scalar(2.0);
        vectorA = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{3});
        rankThreeTensor = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2});
    }

    @Before
    public void enableDebugModeForNaNChecking() {
        TensorValidator.NAN_CATCHER.enable();
        TensorValidator.NAN_FIXER.enable();
    }

    @After
    public void disableDebugModeForNaNChecking() {
        TensorValidator.NAN_CATCHER.disable();
        TensorValidator.NAN_FIXER.disable();
    }

    @Test
    public void youCanCreateARankZeroTensor() {
        DoubleTensor scalar = DoubleTensor.create(new double[]{2.0}, new long[]{});
        DoubleTensor expected = DoubleTensor.scalar(2.0);
        assertEquals(expected, scalar);
        assertEquals(0, scalar.getRank());
    }

    @Test
    public void youCanCreateARankOneTensor() {
        DoubleTensor vector = DoubleTensor.create(new double[]{1, 2, 3, 4, 5}, new long[]{5});
        assertArrayEquals(new long[]{5}, vector.getShape());
    }

    @Test
    public void canAverage() {
        assertEquals(2.5, matrixA.average(), 1e-6);
    }

    @Test
    public void canStandardDeviation() {
        DoubleTensor A = DoubleTensor.create(0, 0.1, -0.1, 0.3, 0.4);

        double actual = A.standardDeviation();
        double expected = 0.20736;

        assertEquals(expected, actual, 1e-3);
    }

    @Test
    public void canStandardDeviationByFormula() {
        KeanuRandom random = new KeanuRandom();
        DoubleTensor A = random.nextGaussian(new long[]{50});

        double actual = A.standardDeviation();
        double expected = Math.sqrt(A.minus(A.average()).pow(2).sum() / (A.getLength() - 1));

        assertEquals(expected, actual, 1e-3);
    }

    @Test
    public void canEye() {
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3);
        DoubleTensor actual = DoubleTensor.eye(3);

        assertEquals(expected, actual);
    }

    @Test
    public void canEye1() {
        DoubleTensor expected = DoubleTensor.create(new double[]{1}, 1, 1);
        DoubleTensor actual = DoubleTensor.eye(1);

        assertEquals(expected, actual);
    }

    @Test
    public void canOnes() {
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 1, 1, 1}, 2, 2);
        DoubleTensor actual = DoubleTensor.ones(2, 2);

        assertEquals(expected, actual);
    }

    @Test
    public void canOnes1() {
        DoubleTensor expected = DoubleTensor.create(1);
        DoubleTensor actual = DoubleTensor.ones(1);
        assertEquals(expected, actual);
    }

    @Test
    public void canOnes0() {
        DoubleTensor expected = DoubleTensor.scalar(1);
        DoubleTensor actual = DoubleTensor.ones();
        assertEquals(expected, actual);
    }

    @Test
    public void canDiagFromVector() {
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3);
        DoubleTensor actual = DoubleTensor.create(1, 2, 3).diag();

        assertEquals(expected, actual);
    }

    @Test
    public void canDiagFromMatrix() {
        DoubleTensor actual = DoubleTensor.create(new double[]{1, 0, 0, 0, 2, 0, 0, 0, 3}, 3, 3).diag();
        DoubleTensor expected = DoubleTensor.create(1, 2, 3);

        assertEquals(expected, actual);
    }

    @Test
    public void canInverseEyeMatrix() {
        DoubleTensor eye = DoubleTensor.eye(2);

        DoubleTensor expected = DoubleTensor.eye(2);
        DoubleTensor actual = eye.matrixInverse();

        assertArrayEquals(expected.asFlatDoubleArray(), actual.asFlatDoubleArray(), 1e-9);
        assertArrayEquals(expected.getShape(), actual.getShape());
    }

    @Test
    public void canInverseMatrix() {
        DoubleTensor A = DoubleTensor.create(1, 2, 3, 4).reshape(2, 2);

        DoubleTensor expected = DoubleTensor.create(4, -2, -3, 1).reshape(2, 2).times(1.0 / A.determinant());

        DoubleTensor actual = A.matrixInverse();

        assertThat(expected, valuesWithinEpsilonAndShapesMatch(actual, 1e-8));
    }

    @Test
    public void canMatrixMultiply() {

        DoubleTensor left = DoubleTensor.create(new double[]{
            1, 2, 3,
            4, 5, 6
        }, 2, 3);

        DoubleTensor right = DoubleTensor.create(new double[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        }, 3, 3);

        DoubleTensor result = left.matrixMultiply(right);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            30, 36, 42,
            66, 81, 96
        }, 2, 3);

        assertEquals(expected, result);
    }

    @Test
    public void canMatrixMultiply2x2() {

        DoubleTensor left = DoubleTensor.create(new double[]{
            1, 2,
            3, 4
        }, 2, 2);

        DoubleTensor right = DoubleTensor.create(new double[]{
            5, 6,
            7, 8
        }, 2, 2);

        DoubleTensor result = left.matrixMultiply(right);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            19, 22,
            43, 50
        }, 2, 2);

        assertEquals(expected, result);
    }

    @Test
    public void canFindDeterminantOf2By2Matrix() {
        DoubleTensor A = DoubleTensor.create(1, 2, 3, 4).reshape(2, 2);
        double expected = 1 * 4 - 2 * 3;
        assertEquals(expected, A.determinant(), 1e-10);
    }

    @Test
    public void canFindDeterminantOfSingular3By3Matrix() {
        DoubleTensor A = DoubleTensor.arange(1, 10).reshape(3, 3);
        double expected = 0;
        assertEquals(expected, A.determinant(), 1e-10);
    }

    @Test
    public void canFindDeterminantOf3By3Matrix() {
        DoubleTensor A = DoubleTensor.create(
            -1, 7, 3,
            -2, -9, 6,
            10, -3, 5
        ).reshape(3, 3);

        double expected = new LUDecomposition(new BlockRealMatrix(new double[][]{
            new double[]{-1, 7, 3},
            new double[]{-2, -9, 6},
            new double[]{10, -3, 5}
        })).getDeterminant();

        assertEquals(expected, A.determinant(), 1e-10);
    }

    @Test
    public void canFindCholeskyDecomposition() {
        //Example from: https://en.wikipedia.org/wiki/Cholesky_decomposition
        DoubleTensor A = DoubleTensor.create(new double[]{
            4, 12, -16,
            12, 37, -43,
            -16, -43, 98
        }, 3, 3);

        DoubleTensor expected = DoubleTensor.create(new double[]{
            2, 0, 0,
            6, 1, 0,
            -8, 5, 3
        }, 3, 3);

        DoubleTensor actual = A.choleskyDecomposition();

        assertThat(actual, valuesWithinEpsilonAndShapesMatch(expected, 1e-10));
    }

    @Test
    public void canElementWiseMultiplyMatrix() {
        DoubleTensor result = matrixA.times(matrixB);
        assertArrayEquals(new double[]{1, 4, 9, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyMatrixByScalar() {
        DoubleTensor result = matrixA.times(scalarA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyScalarByMatrix() {
        DoubleTensor result = scalarA.times(matrixA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canElementWiseMultiplyMatrixInPlace() {
        DoubleTensor result = matrixA.timesInPlace(matrixB);
        assertArrayEquals(new double[]{1, 4, 9, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canMultiplyMatrixByScalarInPlace() {
        DoubleTensor result = matrixA.timesInPlace(scalarA);
        assertArrayEquals(new double[]{2, 4, 6, 8}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canElementWiseDivideMatrix() {
        DoubleTensor result = matrixA.div(matrixB);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0, 1.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canDivideMatrixByScalar() {
        DoubleTensor result = matrixA.div(scalarA);
        assertArrayEquals(new double[]{1.0 / 2.0, 2.0 / 2.0, 3.0 / 2.0, 4.0 / 2.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canDivideScalarByMatrix() {
        DoubleTensor result = scalarA.div(matrixA);
        assertArrayEquals(new double[]{2.0 / 1.0, 2.0 / 2.0, 2.0 / 3.0, 2.0 / 4.0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAMatrix() {
        DoubleTensor mask = matrixA.greaterThanMask(DoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereGreaterThanAScalar() {
        DoubleTensor mask = matrixA.greaterThanMask(DoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{1, 2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAMatrix() {
        DoubleTensor mask = matrixA.lessThanOrEqualToMask(DoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanOrEqualAScalar() {
        DoubleTensor mask = matrixA.lessThanOrEqualToMask(DoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAMatrix() {
        DoubleTensor mask = matrixA.lessThanMask(DoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWhereLessThanAScalar() {
        DoubleTensor mask = matrixA.lessThanMask(DoubleTensor.scalar(2.0));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test

    public void canSetToZero() {
        DoubleTensor mask = matrixA.lessThanMask(DoubleTensor.create(new double[]{2, 2, 2, 2}, new long[]{2, 2}));
        DoubleTensor result = matrixA.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 2, 3, 4}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canTestIfIsNaN() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});
        assertThat(matrix.isNaN(), hasValue(false, false, true, false));
    }

    @Test
    public void canTestIfIsNaNWithScalar() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        assertThat(nan.isNaN(), hasValue(true));
        assertThat(notNan.isNaN(), hasValue(false));
    }

    @Test
    public void canSetWhenNaN() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, -2.0);

        assertArrayEquals(new double[]{-2, -2, -2, -2}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetToZeroWhenNaN() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, Double.NaN, 4}, new long[]{2, 2});

        DoubleTensor mask = DoubleTensor.ones(matrix.getShape());
        DoubleTensor result = matrix.setWithMaskInPlace(mask, 0.0);

        assertArrayEquals(new double[]{0, 0, 0, 0}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void cannotSetIfMaskLengthIsSmallerThanTensorLength() {
        DoubleTensor tensor = DoubleTensor.create(new double[]{1., 2., 3., 4.}, new long[]{2, 2});
        DoubleTensor mask = DoubleTensor.scalar(1.);

        thrown.expect(IllegalArgumentException.class);

        tensor.setWithMaskInPlace(mask, -2.0);
    }

    @Test
    public void cannotSetIfMaskLengthIsLargerThanTensorLength() {
        DoubleTensor tensor = DoubleTensor.create(3);
        DoubleTensor mask = DoubleTensor.ones(2, 2);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("The lengths of the tensor and mask must match, but got tensor length: " + tensor.getLength() + ", mask length: " + mask.getLength());

        tensor.setWithMaskInPlace(mask, -2.0);
    }

    @Test
    public void canApplyUnaryFunctionToScalar() {
        DoubleTensor result = scalarA.apply(a -> a * 2);
        assertEquals(4, result.scalar(), 0.0);
    }

    @Test
    public void canApplyUnaryFunctionToRank3() {
        DoubleTensor rank3Tensor = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2});
        DoubleTensor result = rank3Tensor.apply(a -> a * 2);
        assertArrayEquals(new double[]{2, 4, 6, 8, 10, 12, 14, 16}, result.asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canApplySqrt() {
        DoubleTensor result = scalarA.sqrt();
        assertEquals(Math.sqrt(2.0), result.scalar(), 0.0);
    }

    @Test
    public void canReciprocal() {
        DoubleTensor a = DoubleTensor.create(2, 5);
        assertThat(a.reciprocal(), valuesAndShapesMatch(DoubleTensor.create(1 / 2.0, 1 / 5.0)));
    }

    @Test
    public void canSetAllValues() {
        DoubleTensor rank5 = DoubleTensor.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3
        }, new long[]{2, 2, 2, 2, 2});
        rank5.setAllInPlace(0.0);
        assertAllValuesAre(rank5, 0.0);
        rank5.setAllInPlace(0.8);
        assertAllValuesAre(rank5, 0.8);
    }

    @Test
    public void canElementwiseEqualsAScalarValue() {
        double value = 42.0;
        double otherValue = 42.1;
        DoubleTensor allTheSame = DoubleTensor.create(value, new long[]{2, 3});
        DoubleTensor notAllTheSame = allTheSame.duplicate();
        notAllTheSame.setValue(otherValue, 1, 1);

        assertThat(allTheSame.elementwiseEquals(value).allTrue(), equalTo(true));
        assertThat(notAllTheSame.elementwiseEquals(value), hasValue(true, true, true, true, false, true));
    }

    @Test
    public void canElementwiseEqualsAScalarValueWithScalar() {
        double value = 42.0;
        DoubleTensor tensor = DoubleTensor.create(value);

        assertThat(tensor.elementwiseEquals(value), hasValue(true));
        assertThat(tensor.elementwiseEquals(value + 0.1), hasValue(false));
    }

    @Test
    public void canEqualsWithEpsilon() {
        double[] aData = new double[]{
            1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 7, 5, 8, 6,
            6, 3, 2, 9, 3, 4, 7, 6, 6, 2, 5, 4, 0, 2, 1, 3};
        double[] bData = new double[aData.length];
        for (int i = 0; i < aData.length; i++) {
            if (i % 2 == 0) {
                bData[i] = aData[i] + 0.4;
            } else {
                bData[i] = aData[i] - 0.4;
            }
        }
        double[] cData = bData.clone();
        cData[0] = cData[0] - 1.0;

        DoubleTensor a = DoubleTensor.create(aData, new long[]{2, 2, 2, 2, 2});
        DoubleTensor b = DoubleTensor.create(bData, new long[]{2, 2, 2, 2, 2});
        DoubleTensor c = DoubleTensor.create(cData, new long[]{2, 2, 2, 2, 2});
        assertTrue("equals with epsilon should be true", a.equalsWithinEpsilon(b, 0.5));
        assertTrue("equals with epsilon should be true (inverted order)", b.equalsWithinEpsilon(a, 0.5));
        assertTrue("equals with epsilon should be not true (max delta is 0.4)", !a.equalsWithinEpsilon(b, 0.2));
        assertTrue("equals with epsilon should be not true (max delta is 1.0)", !a.equalsWithinEpsilon(c, 0.5));
    }

    @Test
    public void doesClampTensor() {
        DoubleTensor A = DoubleTensor.create(new double[]{0.25, 3, -4, -5}, new long[]{1, 4});
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(-4.5), DoubleTensor.scalar(2.0));
        DoubleTensor expected = DoubleTensor.create(new double[]{0.25, 2.0, -4.0, -4.5}, new long[]{1, 4});
        assertEquals(expected, clampedA);
    }

    @Test
    public void doesClampScalarWithinBounds() {
        DoubleTensor A = DoubleTensor.scalar(0.25);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 0.25;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }

    @Test
    public void doesClampScalarGreaterThanBounds() {
        DoubleTensor A = DoubleTensor.scalar(5);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 1.0;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }

    @Test
    public void doesClampScalarLessThanBounds() {
        DoubleTensor A = DoubleTensor.scalar(-2);
        DoubleTensor clampedA = A.clamp(DoubleTensor.scalar(0.0), DoubleTensor.scalar(1.0));
        double expected = 0.0;
        assertEquals(expected, clampedA.scalar(), 0.0);
    }

    private void assertAllValuesAre(DoubleTensor tensor, double v) {
        for (double element : tensor.asFlatList()) {
            assertEquals(element, v, 0.01);
        }
    }

    @Test
    public void canTranspose() {
        DoubleTensor a = DoubleTensor.create(new double[]{1, 2, 3, 4}, new long[]{2, 2});
        DoubleTensor actual = a.transpose();
        DoubleTensor expected = DoubleTensor.create(new double[]{1, 3, 2, 4}, 2, 2);

        assertEquals(expected, actual);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotTransposeVector() {
        DoubleTensor.create(1, 2, 3).transpose();
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnIncorrectShape() {
        DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5, 6}, 2, 3);
    }

    @Test
    public void canReshape() {
        DoubleTensor a = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 2, 3);
        DoubleTensor actual = a.reshape(3, 2);
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 3, 2);

        assertEquals(actual, expected);
    }

    @Test
    public void canReshapeWithWildCardDim() {
        DoubleTensor a = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 2, 3);
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 3, 2);

        assertEquals(a.reshape(3, -1), expected);
        assertEquals(a.reshape(-1, 2), expected);
    }

    @Test
    public void canReshapeWithWildCardDimEvenWithLengthOneDim() {
        DoubleTensor a = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 1, 6);
        DoubleTensor expected = DoubleTensor.create(new double[]{0, 1, 2, 3, 4, 5}, 1, 6);

        assertEquals(a.reshape(-1, 6), expected);
    }

    @Test
    public void canPermuteUpperDimensions() {
        DoubleTensor a = DoubleTensor.create(new double[]{
            1, 2,
            3, 4,
            5, 6,
            7, 8
        }, new long[]{1, 2, 2, 2});
        DoubleTensor permuted = a.permute(0, 1, 3, 2);
        DoubleTensor expected = DoubleTensor.create(new double[]{
            1, 3,
            2, 4,
            5, 7,
            6, 8
        }, new long[]{1, 2, 2, 2});

        assertEquals(expected, permuted);
    }

    @Test
    public void canPermute() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{1, 3});
        DoubleTensor y = DoubleTensor.create(new double[]{4, 5, 6}, new long[]{1, 3});

        DoubleTensor concatDimensionZero = DoubleTensor.concat(0, x, y);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatDimensionZero.asFlatDoubleArray(), 1e-6);

        DoubleTensor concatDimensionOne = DoubleTensor.concat(1, x, y);
        DoubleTensor permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, permuttedConcatDimensionOne.asFlatDoubleArray(), 1e-6);

        x = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, new long[]{2, 2, 2});
        y = DoubleTensor.create(new double[]{9, 10, 11, 12, 13, 14, 15, 16}, new long[]{2, 2, 2});

        concatDimensionZero = DoubleTensor.concat(0, x, y);

        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, concatDimensionZero.asFlatDoubleArray(), 1e-6);

        concatDimensionOne = DoubleTensor.concat(1, x, y);
        permuttedConcatDimensionOne = concatDimensionOne.permute(1, 0, 2);

        double[] sliced = new double[permuttedConcatDimensionOne.asFlatDoubleArray().length / 2];
        for (int i = 0; i < permuttedConcatDimensionOne.asFlatDoubleArray().length / 2; i++) {
            sliced[i] = permuttedConcatDimensionOne.asFlatDoubleArray()[i];
        }

        DoubleTensor answer = DoubleTensor.create(sliced, x.getShape()).permute(1, 0, 2);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, answer.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canLinSpace() {
        DoubleTensor actual = DoubleTensor.linspace(0, 10, 5);
        DoubleTensor expected = DoubleTensor.create(0, 2.5, 5.0, 7.5, 10.0);
        assertEquals(expected, actual);
        assertEquals(1, actual.getRank());
    }

    @Test
    public void canARange() {
        DoubleTensor actual = DoubleTensor.arange(0, 5);
        DoubleTensor expected = DoubleTensor.create(0, 1, 2, 3, 4);
        assertEquals(expected, actual);
        assertEquals(1, actual.getRank());
    }

    @Test
    public void canARangeWithStep() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 2);
        DoubleTensor expected = DoubleTensor.create(3, 5);
        assertEquals(expected, actual);
        assertEquals(1, actual.getRank());
    }

    @Test
    public void canARangeWithFractionStep() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 0.5);
        DoubleTensor expected = DoubleTensor.create(3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5);
        assertEquals(expected, actual);
    }

    @Test
    public void canARangeWithFractionStepThatIsNotEvenlyDivisible() {
        DoubleTensor actual = DoubleTensor.arange(3, 7, 1.5);
        DoubleTensor expected = DoubleTensor.create(3.0, 4.5, 6.0);
        assertEquals(expected, actual);
    }

    @Test
    public void canTensorMultiplyWithVectorAndRank4() {
        DoubleTensor a = DoubleTensor.create(new double[]{1, 2, 3}, new long[]{1, 1, 3, 1});
        DoubleTensor b = DoubleTensor.create(new double[]{
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8,
            5, 2, 3, 7, 8
        }, new long[]{1, 3, 1, 5});

        DoubleTensor c = a.tensorMultiply(b, new int[]{2, 3}, new int[]{1, 0});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            30, 12, 18, 42, 48
        }, new long[]{1, 1, 1, 5});

        assertEquals(expected, c);
    }

    @Test
    public void canTensorMultiplyWithNumpyExample() {
        DoubleTensor a = DoubleTensor.arange(0, 60).reshape(3, 4, 5);
        DoubleTensor b = DoubleTensor.arange(0, 24.).reshape(4, 3, 2);
        DoubleTensor c = a.tensorMultiply(b, new int[]{1, 0}, new int[]{0, 1});

        DoubleTensor expected = DoubleTensor.create(new double[]{
            4400., 4730.,
            4532., 4874.,
            4664., 5018.,
            4796., 5162.,
            4928., 5306.
        }, new long[]{5, 2});

        assertEquals(expected, c);
    }

    @Test
    public void canTensorMultiplyAllDimensions() {
        DoubleTensor a = DoubleTensor.create(new double[]{2}).reshape(1);
        DoubleTensor b = DoubleTensor.create(new double[]{1, 2, 3, 4}).reshape(2, 1, 2);
        DoubleTensor resultAB = a.tensorMultiply(b, new int[]{0}, new int[]{1});
        DoubleTensor resultBA = b.tensorMultiply(a, new int[]{1}, new int[]{0});

        assertArrayEquals(new long[]{2, 2}, resultAB.getShape());
        assertArrayEquals(new long[]{2, 2}, resultBA.getShape());
    }

    @Test
    public void canCalculateProductOfVector() {
        double productVectorA = vectorA.product();
        double productRankThreeTensor = rankThreeTensor.product();

        assertEquals(6., productVectorA, 1e-6);
        assertEquals(40320, productRankThreeTensor, 1e-6);
    }

    @Test
    public void scalarMinusInPlaceTensorBehavesSameAsMinus() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.minus(tensor).asFlatDoubleArray(), scalar.minusInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarPlusInPlaceTensorBehavesSameAsPlus() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.plus(tensor).asFlatDoubleArray(), scalar.plusInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarTimesInPlaceTensorBehavesSameAsTimes() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.times(tensor).asFlatDoubleArray(), scalar.timesInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void scalarDivInPlaceTensorBehavesSameAsDiv() {
        DoubleTensor scalar = DoubleTensor.scalar(1);
        DoubleTensor tensor = DoubleTensor.create(2, new long[]{1, 4});

        assertArrayEquals(scalar.div(tensor).asFlatDoubleArray(), scalar.divInPlace(tensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorMinusInPlaceLargerTensorBehavesSameAsMinus() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new long[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.minus(largerTensor).asFlatDoubleArray(), smallerTensor.minusInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorPlusInPlaceLargerTensorBehavesSameAsPlus() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new long[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.plus(largerTensor).asFlatDoubleArray(), smallerTensor.plusInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorTimesInPlaceLargerTensorBehavesSameAsTimes() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new long[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.times(largerTensor).asFlatDoubleArray(), smallerTensor.timesInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorTimesInPlaceLargerTensorBehavesSameAsTimess() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new long[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(largerTensor.times(smallerTensor).asFlatDoubleArray(), largerTensor.timesInPlace(smallerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void smallerTensorDivInPlaceLargerTensorBehavesSameAsDiv() {
        DoubleTensor smallerTensor = DoubleTensor.create(2, new long[]{2, 2});
        DoubleTensor largerTensor = DoubleTensor.create(3, new long[]{2, 2, 2});

        assertArrayEquals(smallerTensor.div(largerTensor).asFlatDoubleArray(), smallerTensor.divInPlace(largerTensor).asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void doesCompareGreaterThanOrEqualScalarTensor() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1., 2., 3., 4.}, new long[]{2, 2});
        BooleanTensor result = matrix.greaterThanOrEqual(DoubleTensor.scalar(3.));
        Boolean[] expected = new Boolean[]{false, false, true, true};
        assertArrayEquals(expected, result.asFlatArray());
    }

    @Test
    public void canSplit() {

        int dim = 2;
        DoubleTensor A = DoubleTensor.arange(0, 24).reshape(2, 3, 1, 4);
        DoubleTensor B = DoubleTensor.arange(24, 96).reshape(2, 3, 3, 4);
        DoubleTensor C = DoubleTensor.arange(96, 144).reshape(2, 3, 2, 4);

        DoubleTensor D = DoubleTensor.concat(dim, A, B, C);
        List<DoubleTensor> splitTensor = D.split(dim, new long[]{1, 4, 6});

        DoubleTensor[] concatList = new DoubleTensor[]{A, B, C};
        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(concatList[i], splitTensor.get(i));
        }

    }

    @Test
    public void canSplitHighRank() {
        assertCanSplit(new long[]{2, 3, 4, 5, 7, 2}, new int[]{3, 2, 6}, 1);
    }

    @Test
    public void canSplitEndDimension() {
        assertCanSplit(new long[]{2, 3, 4, 5}, new int[]{3, 4, 2}, 3);
    }

    @Test
    public void canSplitFirstDimension() {
        assertCanSplit(new long[]{2, 3, 4, 5, 7, 2}, new int[]{3, 4, 2, 6, 9, 2}, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnZeroLengthSplit() {
        DoubleTensor A = DoubleTensor.arange(0, 100).reshape(10, 10);
        A.split(0, new long[]{0});
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidNegativeDimensionSplit() {
        DoubleTensor A = DoubleTensor.arange(0, 100).reshape(10, 10);
        A.split(-3, new long[]{1, 5});
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidDimensionSplit() {
        DoubleTensor A = DoubleTensor.arange(0, 100).reshape(10, 10);
        A.split(3, new long[]{1, 5});
    }

    @Test
    public void doesSatisfyJavaDocExample() {
        DoubleTensor A = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3}, 2, 6);

        List<DoubleTensor> actual = A.split(1, new long[]{1, 3, 6});

        DoubleTensor expected0 = DoubleTensor.create(new double[]{1, 7}, 2, 1);
        DoubleTensor expected1 = DoubleTensor.create(new double[]{2, 3, 8, 9}, 2, 2);
        DoubleTensor expected2 = DoubleTensor.create(new double[]{4, 5, 6, 1, 2, 3}, 2, 3);

        assertThat(expected0, valuesAndShapesMatch(actual.get(0)));
        assertThat(expected1, valuesAndShapesMatch(actual.get(1)));
        assertThat(expected2, valuesAndShapesMatch(actual.get(2)));
    }

    @Test
    public void canFindScalarMinAndMax() {
        DoubleTensor a = DoubleTensor.create(5., 4., 3., 2.).reshape(2, 2);
        double min = a.min();
        double max = a.max();
        assertEquals(2., min, 1e-6);
        assertEquals(5., max, 1e-6);
    }

    @Test
    public void canFindMinAndMaxFromScalarToTensor() {
        DoubleTensor a = DoubleTensor.create(5., 4., 3., 2.).reshape(1, 4);
        DoubleTensor b = DoubleTensor.scalar(3.);

        DoubleTensor min = DoubleTensor.min(a, b);
        DoubleTensor max = DoubleTensor.max(a, b);

        assertArrayEquals(new double[]{3, 3, 3, 2}, min.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(new double[]{5, 4, 3, 3}, max.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canFindMinFromScalarToTensorInPlace() {
        DoubleTensor a = DoubleTensor.create(5., 4., 3., 2.).reshape(1, 4);
        DoubleTensor b = DoubleTensor.scalar(3.);

        a.minInPlace(b);

        assertArrayEquals(new double[]{3, 3, 3, 2}, a.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canFindMaxFromScalarToTensorInPlace() {
        DoubleTensor a = DoubleTensor.create(5., 4., 3., 2.).reshape(1, 4);
        DoubleTensor b = DoubleTensor.scalar(3.);

        a.maxInPlace(b);

        assertArrayEquals(new double[]{5, 4, 3, 3}, a.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canFindElementWiseMinAndMax() {
        DoubleTensor a = DoubleTensor.create(1., 2., 3., 4.).reshape(1, 4);
        DoubleTensor b = DoubleTensor.create(2., 3., 1., 4.).reshape(1, 4);

        DoubleTensor min = DoubleTensor.min(a, b);
        DoubleTensor max = DoubleTensor.max(a, b);

        assertArrayEquals(new double[]{1, 2, 1, 4}, min.asFlatDoubleArray(), 1e-6);
        assertArrayEquals(new double[]{2, 3, 3, 4}, max.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canFindArgMaxOfRowVector() {
        DoubleTensor tensorRow = DoubleTensor.create(1, 3, 4, 5, 2).reshape(1, 5);

        assertEquals(3, tensorRow.argMax());
        assertThat(tensorRow.argMax(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMax(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
    }

    @Test
    public void canFindArgMaxOfColumnVector() {
        DoubleTensor tensorCol = DoubleTensor.create(1, 3, 4, 5, 2).reshape(5, 1);

        assertEquals(3, tensorCol.argMax());
        assertThat(tensorCol.argMax(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{3}, 1)));
        assertThat(tensorCol.argMax(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void argMaxReturnsIndexOfFirstMax() {
        DoubleTensor tensor = DoubleTensor.create(1, 5, 5, 5, 5);

        assertEquals(tensor.argMax(), 1);
    }

    @Test
    public void canFindArgMaxOfMatrix() {
        DoubleTensor tensor = DoubleTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0, 0, 0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 0)));
        assertEquals(2, tensor.argMax());
    }

    @Test
    public void canFindArgMaxNumPyExample() {
        /*
         * >>> a = np.arange(6).reshape(2,3) + 10
         * >>> a
         * array([[10, 11, 12],
         *        [13, 14, 15]])
         * >>> np.argmax(a)
         * 5
         * >>> np.argmax(a, axis=0)
         * array([1, 1, 1])
         * >>> np.argmax(a, axis=1)
         * array([2, 2])
         */

        DoubleTensor a = DoubleTensor.arange(0, 6).reshape(2, 3).plus(10);

        assertEquals(5, a.argMax());
        assertThat(a.argMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 1, 1)));
        assertThat(a.argMax(1), valuesAndShapesMatch(IntegerTensor.create(2, 2)));
    }

    @Test
    public void canFindNanArgMaxOfMatrixNumPyExample() {

        /*
         * >>> a = np.array([[np.nan, 4], [2, 3]])
         * >>> np.argmax(a)
         * 0
         * >>> np.nanargmax(a)
         * 1
         * >>> np.nanargmax(a, axis=0)
         * array([1, 0])
         * >>> np.nanargmax(a, axis=1)
         * array([1, 1])
         */

        DoubleTensor tensor = DoubleTensor.create(Double.NaN, 4, 2, 3).reshape(2, 2);

        assertEquals(0, tensor.argMax());
        assertEquals(1, tensor.nanArgMax());
        assertThat(tensor.nanArgMax(0), valuesAndShapesMatch(IntegerTensor.create(1, 0)));
        assertThat(tensor.nanArgMax(1), valuesAndShapesMatch(IntegerTensor.create(1, 1)));
    }

    @Test
    public void canFindArgMaxOfHighRank() {
        DoubleTensor tensor = DoubleTensor.arange(0, 512).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.ones(8, 4, 2, 4)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.create(7, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMax(2), valuesAndShapesMatch(IntegerTensor.create(3, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMax(3), valuesAndShapesMatch(IntegerTensor.ones(2, 8, 4, 4)));
        assertEquals(511, tensor.argMax());
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHigh() {
        DoubleTensor tensor = DoubleTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMax(2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMaxFailsForAxisTooHighWithScalar() {
        DoubleTensor tensor = DoubleTensor.scalar(1);
        tensor.argMax(2);
    }

    @Test
    public void canFindArgMinOfRowVector() {
        DoubleTensor tensorRow = DoubleTensor.create(7, 3, 4, 5, 2).reshape(1, 5);

        assertEquals(4, tensorRow.argMin());
        assertThat(tensorRow.argMin(0), valuesAndShapesMatch(IntegerTensor.zeros(5)));
        assertThat(tensorRow.argMin(1), valuesAndShapesMatch(IntegerTensor.create(new int[]{4}, 1)));
    }

    @Test
    public void canFindArgMinOfColumnVector() {
        DoubleTensor tensorCol = DoubleTensor.create(7, 1, 4, 5, 2).reshape(5, 1);

        assertEquals(1, tensorCol.argMin());
        assertThat(tensorCol.argMin(0), valuesAndShapesMatch(IntegerTensor.create(new int[]{1}, 1)));
        assertThat(tensorCol.argMin(1), valuesAndShapesMatch(IntegerTensor.zeros(5)));
    }

    @Test
    public void argMinReturnsIndexOfFirstMin() {
        DoubleTensor tensor = DoubleTensor.create(5, 2, 2, 2, 2);

        assertEquals(1, tensor.argMin());
    }

    @Test
    public void canFindArgMinOfMatrix() {
        DoubleTensor tensor = DoubleTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);

        assertThat(tensor.argMin(0), valuesAndShapesMatch(IntegerTensor.create(0, 1, 1, 1)));
        assertThat(tensor.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, 1)));
        assertEquals(0, tensor.argMin());
    }

    @Test
    public void canFindArgMinOfHighRank() {
        DoubleTensor tensor = DoubleTensor.arange(0, 512).reshape(2, 8, 4, 2, 4);

        assertThat(tensor.argMin(0), valuesAndShapesMatch(IntegerTensor.zeros(8, 4, 2, 4)));
        assertThat(tensor.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, new long[]{2, 4, 2, 4})));
        assertThat(tensor.argMin(2), valuesAndShapesMatch(IntegerTensor.create(0, new long[]{2, 8, 2, 4})));
        assertThat(tensor.argMin(3), valuesAndShapesMatch(IntegerTensor.zeros(2, 8, 4, 4)));
        assertEquals(0, tensor.argMin());
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMinFailsForAxisTooHigh() {
        DoubleTensor tensor = DoubleTensor.create(1, 2, 4, 3, 3, 1, 3, 1).reshape(2, 4);
        tensor.argMin(2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void argMinFailsForAxisTooHighWithScalar() {
        DoubleTensor tensor = DoubleTensor.scalar(1);
        tensor.argMin(2);
    }

    @Test
    public void canArgMinNumPyExample() {

        /*
         * >>> a = np.arange(6).reshape(2,3) + 10
         * >>> a
         * array([[10, 11, 12],
         *        [13, 14, 15]])
         * >>> np.argmin(a)
         * 0
         * >>> np.argmin(a, axis=0)
         * array([0, 0, 0])
         * >>> np.argmin(a, axis=1)
         * array([0, 0])
         */

        DoubleTensor a = DoubleTensor.arange(0, 6).reshape(2, 3).plus(10);

        assertEquals(0, a.argMin());
        assertThat(a.argMin(0), valuesAndShapesMatch(IntegerTensor.create(0, 0, 0)));
        assertThat(a.argMin(1), valuesAndShapesMatch(IntegerTensor.create(0, 0)));
    }

    @Test
    public void canFindNanArgMinNumPyExample() {

        /*
         * >>> a = np.array([[np.nan, 4], [2, 3]])
         * >>> np.argmin(a)
         * 0
         * >>> np.nanargmin(a)
         * 2
         * >>> np.nanargmin(a, axis=0)
         * array([1, 1])
         * >>> np.nanargmin(a, axis=1)
         * array([1, 0])
         */

        DoubleTensor tensor = DoubleTensor.create(Double.NaN, 4, 2, 3).reshape(2, 2);

        assertEquals(0, tensor.argMin());
        assertEquals(2, tensor.nanArgMin());
        assertThat(tensor.nanArgMin(0), valuesAndShapesMatch(IntegerTensor.create(1, 1)));
        assertThat(tensor.nanArgMin(1), valuesAndShapesMatch(IntegerTensor.create(1, 0)));
    }

    private void assertCanSplit(long[] baseShape, int[] concatenatedIndices, int concatenatedDimension) {

        long[] splitIndices = new long[concatenatedIndices.length];
        List<DoubleTensor> toConcat = new ArrayList<>();

        long previousEndLength = 0;
        long splitPosition = 0;
        for (int i = 0; i < concatenatedIndices.length; i++) {
            long[] shape = Arrays.copyOf(baseShape, baseShape.length);
            shape[concatenatedDimension] = concatenatedIndices[i];

            splitIndices[i] = splitPosition + concatenatedIndices[i];
            splitPosition = splitIndices[i];

            long newEndLength = previousEndLength + TensorShape.getLength(shape);
            toConcat.add(DoubleTensor.arange(previousEndLength, newEndLength).reshape(shape));
            previousEndLength = newEndLength;
        }

        DoubleTensor D = DoubleTensor.concat(concatenatedDimension, toConcat.toArray(new DoubleTensor[toConcat.size() - 1]));
        List<DoubleTensor> splitTensor = D.split(concatenatedDimension, splitIndices);

        for (int i = 0; i < splitTensor.size(); i++) {
            assertEquals(toConcat.get(i), splitTensor.get(i));
        }
    }

    @Test
    public void youCanCheckForZeros() {
        DoubleTensor containsZero = DoubleTensor.create(new double[]{
                0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
                Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0},
            4, 2);

        BooleanTensor expectedMask = BooleanTensor.create(new boolean[]{
                false, true, true, true,
                true, true, true, false},
            4, 2);

        assertThat(TensorValidator.ZERO_CATCHER.check(containsZero), equalTo(expectedMask));
    }

    @Test
    public void youCanFixAValidationIssueByReplacingANaN() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.NAN_FIXER;
        nan = validator.validate(nan);
        notNan = validator.validate(notNan);
        assertThat(nan, equalTo(zero));
        assertThat(notNan, equalTo(notNan));
    }

    @Test
    public void youCanCheckForNaNs() {
        DoubleTensor containsNan = DoubleTensor.create(new double[]{
                0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
                Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0},
            4, 2);

        BooleanTensor expectedMask = BooleanTensor.create(new boolean[]{
                true, true, true, false,
                true, true, true, true},
            4, 2);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.NAN_CATCHER;
        assertThat(validator.check(containsNan), equalTo(expectedMask));
        assertThat(containsNan.isNaN(), equalTo(expectedMask.not()));
    }

    @Test
    public void youCanReplaceNaNs() {
        double[] input = {
            0.0, -1.0, -Double.NEGATIVE_INFINITY, Double.NaN,
            Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0};

        Double[] expectedOutput = {
            0.0, -1.0, -Double.NEGATIVE_INFINITY, 0.0,
            Double.POSITIVE_INFINITY, Double.MIN_VALUE, Double.MAX_VALUE, -0.0};

        DoubleTensor containsNan = DoubleTensor.create(input,
            4, 2);

        assertThat(containsNan.replaceNaN(0.), hasValue(expectedOutput));
    }

    @Test
    public void youCanReplaceNaNsWithScalar() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        assertThat(nan.replaceNaN(0.), hasValue(0.));
        assertThat(notNan.replaceNaN(0.), hasValue(Double.NEGATIVE_INFINITY));
    }

    @Test
    public void canSetWhenNaNWithScalar() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        assertThat(nan.setWithMaskInPlace(mask, 2.), hasValue(2.));
    }

    @Test
    public void canSetToZeroWhenNaNWithScalar() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor mask = DoubleTensor.scalar(1.);
        assertThat(nan.setWithMaskInPlace(mask, 0.), hasValue(0.));
    }

    @Test
    public void youCanCheckForZerosWithScalar() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor nonZero = DoubleTensor.scalar(1e-8);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.ZERO_CATCHER;
        assertThat(validator.check(zero), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(nonZero), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanCheckForNans() {
        DoubleTensor nan = DoubleTensor.scalar(Double.NaN);
        DoubleTensor notNan = DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.NAN_CATCHER;
        assertThat(nan.isNaN(), equalTo(BooleanTensor.scalar(true)));
        assertThat(notNan.isNaN(), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(nan), equalTo(BooleanTensor.scalar(false)));
        assertThat(validator.check(notNan), equalTo(BooleanTensor.scalar(true)));
    }

    @Test
    public void youCanDoYLogXEvenWhenBothAreZero() {
        DoubleTensor x = DoubleTensor.create(
            Double.MIN_VALUE, 1e-8, 1., 1e8);
        DoubleTensor y = x.duplicate();
        assertThat(x.log().times(y), equalTo(x.safeLogTimes(y)));

        DoubleTensor zeros = DoubleTensor.create(0., -0.);

        assertThat(zeros.safeLogTimes(zeros), hasValue(0., 0.));
        assertThat(zeros.log().times(zeros), hasValue(Double.NaN, Double.NaN));
    }

    @Test
    public void youCanDoYLogXEvenWhenBothAreZeroWithScalar() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        assertThat(zero.safeLogTimes(zero), hasValue(0.));
        assertThat(zero.log().times(zero), hasValue(Double.NaN));
    }

    @Test
    public void logTimesFailsIfYouPassInATensorThatAlreadyContainsNaN() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.create(1., 1.);
        DoubleTensor y = DoubleTensor.create(1., Double.NaN);
        x.safeLogTimes(y);
    }

    @Test
    public void logTimesFailsIfYouPassInATensorThatAlreadyContainsNaNWithScalar() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.scalar(1.);
        DoubleTensor y = DoubleTensor.scalar(Double.NaN);
        x.safeLogTimes(y);
    }

    @Test
    public void logTimesFailsIfYouStartWithATensorThatAlreadyContainsNaN() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.create(1., Double.NaN);
        DoubleTensor y = DoubleTensor.create(1., 1.);
        x.safeLogTimes(y);
    }

    @Test
    public void logTimesFailsIfYouStartWithATensorThatAlreadyContainsNaNWithScalar() {
        expectedException.expect(TensorValueException.class);
        expectedException.expectMessage("Invalid value found");

        DoubleTensor x = DoubleTensor.scalar(Double.NaN);
        DoubleTensor y = DoubleTensor.scalar(1.);
        x.safeLogTimes(y);
    }

    @Test
    public void youCanFixAValidationIssueByReplacingTheValue() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, -1.0);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatReplaces(0., 1e-8);
        DoubleTensor actual = validator.validate(containsZero);
        assertThat(actual, valuesAndShapesMatch(expectedResult));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValue() {
        DoubleTensor containsZero = DoubleTensor.create(1.0, 0.0, -1.0);
        DoubleTensor expectedResult = DoubleTensor.create(1.0, 1e-8, 1e-8);

        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatFixesElementwise(x -> x > 0., (TensorValidationPolicy<Double, DoubleTensor>) TensorValidationPolicy.changeValueTo(1e-8));
        DoubleTensor actual = validator.validate(containsZero);
        assertThat(actual, equalTo(expectedResult));
    }

    @Test
    public void youCanFixACustomValidationIssueByReplacingTheValueWithScalar() {
        DoubleTensor tensor1 = DoubleTensor.scalar(0.);
        DoubleTensor tensor2 = DoubleTensor.scalar(1.);
        DoubleTensor one = DoubleTensor.scalar(1.);
        DoubleTensor notZero = DoubleTensor.scalar(1e-8);
        Function<Double, Boolean> checkFunction = x -> x > 0.;
        TensorValidator<Double, DoubleTensor> validator = TensorValidator.thatFixesElementwise(checkFunction, (TensorValidationPolicy<Double, DoubleTensor>) TensorValidationPolicy.changeValueTo(1e-8));
        tensor1 = validator.validate(tensor1);
        tensor2 = validator.validate(tensor2);
        assertThat(tensor1, equalTo(notZero));
        assertThat(tensor2, equalTo(one));
    }

    @Test
    public void comparesDoubleTensorWithScalar() {
        DoubleTensor value = DoubleTensor.create(1., 2., 3.);
        DoubleTensor differentValue = DoubleTensor.scalar(1.);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void canSumOverSpecifiedDimensionOfRank3() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        DoubleTensor summation = x.sum(2);
        DoubleTensor expected = DoubleTensor.create(new double[]{3, 7, 11, 15}, 2, 2);
        assertThat(summation, valuesAndShapesMatch(expected));
    }

    @Test
    public void canSumOverSpecifiedDimensionOfMatrix() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        DoubleTensor summationRow = x.sum(1);
        DoubleTensor expected = DoubleTensor.create(3, 7);
        assertThat(summationRow, valuesAndShapesMatch(expected));
    }

    @Test
    public void canSumOverSpecifiedDimensionOfVector() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4);
        DoubleTensor summation = x.sum(0);
        DoubleTensor expected = DoubleTensor.scalar(10);
        assertThat(summation, valuesAndShapesMatch(expected));
    }

    @Test
    public void canProductOverSpecifiedDimensionOfRank3() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        assertThat(x.product(0), valuesAndShapesMatch(DoubleTensor.create(new double[]{5, 12, 21, 32}, 2, 2)));
        assertThat(x.product(1), valuesAndShapesMatch(DoubleTensor.create(new double[]{3, 8, 35, 48}, 2, 2)));
        assertThat(x.product(2), valuesAndShapesMatch(DoubleTensor.create(new double[]{2, 12, 30, 56}, 2, 2)));
    }

    @Test
    public void canProductOverSpecifiedDimensionOfMatrix() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2);
        assertThat(x.product(0), valuesAndShapesMatch(DoubleTensor.create(3, 8)));
        assertThat(x.product(1), valuesAndShapesMatch(DoubleTensor.create(2, 12)));
    }

    @Test
    public void canProductOverSpecifiedDimensionOfVector() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4);
        assertThat(x.product(0), valuesAndShapesMatch(DoubleTensor.scalar(24)));
    }

    @Test
    public void canDuplicateRank1() {
        DoubleTensor x = DoubleTensor.create(1, 2);
        assertEquals(x, x.duplicate());
    }

    @Test
    public void canDuplicateRank0() {
        DoubleTensor x = DoubleTensor.scalar(1.0);
        assertEquals(x, x.duplicate());
    }

    @Test
    public void doesDownRankOnSliceRank3To2() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4, 1, 2, 3, 4).reshape(2, 2, 2);
        TensorTestHelper.doesDownRankOnSliceRank3To2(x);
    }

    @Test
    public void doesDownRankOnSliceRank2To1() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4).reshape(2, 2);
        TensorTestHelper.doesDownRankOnSliceRank2To1(x);
    }

    @Test
    public void canSliceRank2() {
        DoubleTensor x = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        DoubleTensor slice = x.slice(1, 0);
        assertThat(slice, valuesAndShapesMatch(DoubleTensor.create(1, 4, 7)));
        assertThat(x, valuesAndShapesMatch(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)));
    }

    @Test
    public void doesDownRankOnSliceRank1ToScalar() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4);
        TensorTestHelper.doesDownRankOnSliceRank1ToScalar(x);
    }

    @Test
    public void canSliceRank1() {
        DoubleTensor x = DoubleTensor.create(1, 2, 3, 4);
        DoubleTensor slice = x.slice(0, 1);
        assertThat(slice, valuesAndShapesMatch(DoubleTensor.scalar(2)));
    }

    @Test
    public void canConcatScalars() {
        DoubleTensor x = DoubleTensor.scalar(2);
        DoubleTensor y = DoubleTensor.scalar(3);

        DoubleTensor concat = DoubleTensor.concat(x, y);
        assertEquals(DoubleTensor.create(2, 3), concat);
    }

    @Test
    public void canConcatVectors() {
        DoubleTensor x = DoubleTensor.create(2, 3);
        DoubleTensor y = DoubleTensor.create(4, 5);

        DoubleTensor concat = DoubleTensor.concat(x, y);
        assertEquals(DoubleTensor.create(2, 3, 4, 5), concat);
    }

    @Test
    public void canConcatMatrices() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5).reshape(1, 2);

        DoubleTensor concat = DoubleTensor.concat(0, x, y);
        assertEquals(DoubleTensor.create(2, 3, 4, 5).reshape(2, 2), concat);
    }

    @Test
    public void canStackScalars() {
        DoubleTensor x = DoubleTensor.scalar(2);
        DoubleTensor y = DoubleTensor.scalar(3);

        assertThat(DoubleTensor.create(2, 3).reshape(2), TensorMatchers.valuesAndShapesMatch(DoubleTensor.stack(0, x, y)));
    }

    @Test
    public void canStackVectors() {
        DoubleTensor x = DoubleTensor.create(2, 3);
        DoubleTensor y = DoubleTensor.create(4, 5);

        assertEquals(DoubleTensor.create(2, 3, 4, 5).reshape(2, 2), DoubleTensor.stack(0, x, y));
        assertEquals(DoubleTensor.create(2, 4, 3, 5).reshape(2, 2), DoubleTensor.stack(1, x, y));
    }

    @Test
    public void canStackMatrices() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5).reshape(1, 2);

        assertThat(DoubleTensor.create(2, 3, 4, 5).reshape(2, 1, 2), valuesAndShapesMatch(DoubleTensor.stack(0, x, y)));
        assertThat(DoubleTensor.create(2, 3, 4, 5).reshape(1, 2, 2), valuesAndShapesMatch(DoubleTensor.stack(1, x, y)));
        /*
        Result in numpy when dimension is equal to array length:
        >>> a
        array([[2, 3]])
        >>> b
        array([[4, 5]])
        >>> np.stack([a, b], axis=2)
        array([[[2, 4],
                [3, 5]]])
        */
        assertThat(DoubleTensor.create(2, 4, 3, 5).reshape(1, 2, 2), valuesAndShapesMatch(DoubleTensor.stack(2, x, y)));
    }

    @Test
    public void canStackIfDimensionIsNegative() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5).reshape(1, 2);

        assertThat(DoubleTensor.create(2, 3, 4, 5).reshape(2, 1, 2), valuesAndShapesMatch(DoubleTensor.stack(-3, x, y)));
        assertThat(DoubleTensor.create(2, 3, 4, 5).reshape(1, 2, 2), valuesAndShapesMatch(DoubleTensor.stack(-2, x, y)));
        assertThat(DoubleTensor.create(2, 4, 3, 5).reshape(1, 2, 2), valuesAndShapesMatch(DoubleTensor.stack(-1, x, y)));
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotStackIfPositiveDimensionIsOutOfBounds() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5).reshape(1, 2);
        DoubleTensor.stack(3, x, y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotStackIfNegativeDimensionIsOutOfBounds() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5).reshape(1, 2);
        DoubleTensor.stack(-4, x, y);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsWhenNeedsDimensionSpecifiedForConcat() {
        DoubleTensor x = DoubleTensor.create(2, 3).reshape(1, 2);
        DoubleTensor y = DoubleTensor.create(4, 5, 6).reshape(1, 3);

        DoubleTensor concat = DoubleTensor.concat(0, x, y);
        assertEquals(DoubleTensor.create(2, 3, 4, 5, 6), concat);
    }

    @Test
    public void canBasicTrig() {
        assertUnaryOperation(FastMath::sin, DoubleTensor::sin, tensorBetween0And1());
        assertUnaryOperation(FastMath::sin, DoubleTensor::sinInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::cos, DoubleTensor::cos, tensorBetween0And1());
        assertUnaryOperation(FastMath::cos, DoubleTensor::cosInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::tan, DoubleTensor::tan, tensorBetween0And1());
        assertUnaryOperation(FastMath::tan, DoubleTensor::tanInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::asin, DoubleTensor::asin, tensorBetween0And1());
        assertUnaryOperation(FastMath::asin, DoubleTensor::asinInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::acos, DoubleTensor::acos, tensorBetween0And1());
        assertUnaryOperation(FastMath::acos, DoubleTensor::acosInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::atan, DoubleTensor::atan, tensorBetween0And1());
        assertUnaryOperation(FastMath::atan, DoubleTensor::atanInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::sinh, DoubleTensor::sinh, tensorBetween0And1());
        assertUnaryOperation(FastMath::sinh, DoubleTensor::sinhInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::cosh, DoubleTensor::cosh, tensorBetween0And1());
        assertUnaryOperation(FastMath::cosh, DoubleTensor::coshInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::tanh, DoubleTensor::tanh, tensorBetween0And1());
        assertUnaryOperation(FastMath::tanh, DoubleTensor::tanhInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::asinh, DoubleTensor::asinh, tensorBetween0And1());
        assertUnaryOperation(FastMath::asinh, DoubleTensor::asinhInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::acosh, DoubleTensor::acosh, tensorBetween(2, 3));
        assertUnaryOperation(FastMath::acosh, DoubleTensor::acoshInPlace, tensorBetween(2, 3));
        assertUnaryOperation(FastMath::atanh, DoubleTensor::atanh, tensorBetween0And1());
        assertUnaryOperation(FastMath::atanh, DoubleTensor::atanhInPlace, tensorBetween0And1());
    }

    @Test
    public void canAtan2() {
        DoubleTensor a = DoubleTensor.create(0.5, 0.2);
        DoubleTensor b = DoubleTensor.create(0.3, 0.4);
        DoubleTensor expected = DoubleTensor.create(FastMath.atan2(0.3, 0.5), FastMath.atan2(0.4, 0.2));
        assertThat(a.atan2(b), valuesAndShapesMatch(expected));
    }

    private DoubleTensor tensorBetween0And1() {
        return DoubleTensor.linspace(0.1, 0.9, 4).reshape(2, 2);
    }

    private DoubleTensor tensorBetween(double start, double end) {
        return DoubleTensor.linspace(start, end, 4).reshape(2, 2);
    }

    @Test
    public void canBasicUnaryOps() {
        assertUnaryOperation(Math::exp, DoubleTensor::exp, tensorRangeWithNegatives());
        assertUnaryOperation(Math::exp, DoubleTensor::expInPlace, tensorRangeWithNegatives());
        assertUnaryOperation(Math::abs, DoubleTensor::abs, tensorRangeWithNegatives());
        assertUnaryOperation(Math::abs, DoubleTensor::absInPlace, tensorRangeWithNegatives());
        assertUnaryOperation(Gamma::digamma, DoubleTensor::digamma, tensorRangeWithNegatives());
        assertUnaryOperation(Gamma::digamma, DoubleTensor::digammaInPlace, tensorRangeWithNegatives());
        assertUnaryOperation(Math::ceil, DoubleTensor::ceil, tensorRangeWithNegatives());
        assertUnaryOperation(Math::ceil, DoubleTensor::ceilInPlace, tensorRangeWithNegatives());
        assertUnaryOperation(Math::floor, DoubleTensor::floor, tensorRangeWithNegatives());
        assertUnaryOperation(Math::floor, DoubleTensor::floorInPlace, tensorRangeWithNegatives());
        assertUnaryOperation(Gamma::logGamma, DoubleTensor::logGamma, tensorBetween0And1());
        assertUnaryOperation(Gamma::logGamma, DoubleTensor::logGammaInPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::sqrt, DoubleTensor::sqrt, tensorBetween0And1());
        assertUnaryOperation(FastMath::sqrt, DoubleTensor::sqrtInPlace, tensorBetween0And1());

        assertUnaryOperation(FastMath::expm1, DoubleTensor::expM1, tensorBetween0And1());
        assertUnaryOperation(FastMath::expm1, DoubleTensor::expM1InPlace, tensorBetween0And1());
        assertUnaryOperation(DoubleTensorTest::exp2, DoubleTensor::exp2, tensorBetween0And1());
        assertUnaryOperation(DoubleTensorTest::exp2, DoubleTensor::exp2InPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::log10, DoubleTensor::log10, tensorBetween0And1());
        assertUnaryOperation(FastMath::log10, DoubleTensor::log10InPlace, tensorBetween0And1());
        assertUnaryOperation(DoubleTensorTest::log2, DoubleTensor::log2, tensorBetween0And1());
        assertUnaryOperation(DoubleTensorTest::log2, DoubleTensor::log2InPlace, tensorBetween0And1());
        assertUnaryOperation(FastMath::log1p, DoubleTensor::log1p, tensorBetween0And1());
        assertUnaryOperation(FastMath::log1p, DoubleTensor::log1pInPlace, tensorBetween0And1());
    }

    public static double exp2(double a) {
        return FastMath.pow(2, a);
    }

    public static double log2(double a) {
        return FastMath.log(a) / FastMath.log(2);
    }

    @Test
    public void canLogAddExp() {
        /*
         * >>> prob1 = np.log(1e-50)
         * >>> prob2 = np.log(2.5e-50)
         * >>> prob12 = np.logaddexp(prob1, prob2)
         * >>> prob12
         * -113.87649168120691
         * >>> np.exp(prob12)
         * 3.5000000000000057e-50
         */

        DoubleTensor prob1 = DoubleTensor.scalar(Math.log(1e-50));
        DoubleTensor prob2 = DoubleTensor.scalar(Math.log(2.5e-50));
        DoubleTensor prob12 = prob1.logAddExp(prob2);
        assertThat(prob12.scalar(), closeTo(-113.87649168120691, 1e-8));
    }

    @Test
    public void canLogAddExp2() {

        /*
         * >>> prob1 = np.log2(1e-50)
         * >>> prob2 = np.log2(2.5e-50)
         * >>> prob12 = np.logaddexp2(prob1, prob2)
         * >>> prob1, prob2, prob12
         * (-166.09640474436813, -164.77447664948076, -164.28904982231052)
         * >>> 2**prob12
         * 3.4999999999999914e-50
         */

        DoubleTensor prob1 = DoubleTensor.scalar(Math.log(1e-50) / Math.log(2));
        DoubleTensor prob2 = DoubleTensor.scalar(Math.log(2.5e-50) / Math.log(2));
        DoubleTensor prob12 = prob1.logAddExp2(prob2);
        assertThat(prob12.scalar(), closeTo(-164.28904982231052, 1e-8));
    }

    private DoubleTensor tensorRangeWithNegatives() {
        return DoubleTensor.linspace(-0.9, 0.9, 4).reshape(2, 2);
    }

    @Test
    public void canSigmoid() {
        final Sigmoid sigmoid = new Sigmoid();
        assertUnaryOperation(sigmoid::value, DoubleTensor::sigmoid, DoubleTensor.arange(1, 5).reshape(2, 2));
        assertUnaryOperation(sigmoid::value, DoubleTensor::sigmoidInPlace, DoubleTensor.arange(1, 5).reshape(2, 2));
    }

    private void assertUnaryOperation(Function<Double, Double> unaryOp, Function<DoubleTensor, DoubleTensor> tensorOp, DoubleTensor input) {

        double[] expectedBuffer = new double[Ints.checkedCast(input.getLength())];
        double[] inputBuffer = input.asFlatDoubleArray();

        for (int i = 0; i < expectedBuffer.length; i++) {
            expectedBuffer[i] = unaryOp.apply(inputBuffer[i]);
        }

        DoubleTensor output = tensorOp.apply(input);
        DoubleTensor expected = DoubleTensor.create(expectedBuffer, input.getShape());

        assertTrue(expected.equalsWithinEpsilon(output, 1e-6));
    }

    @Test
    public void canGetValueByIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        double value = A.getValue(1, 0, 1);
        assertEquals(5, value, 1e-10);
    }

    @Test
    public void canGetValueByFlatIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        double value = A.getValue(7);
        assertEquals(7, value, 1e-10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnGetInvalidIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        double value = A.getValue(1, 0);
    }

    @Test
    public void canSetValueByFlatIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        A.setValue(0.5, 6);
        assertEquals(0.5, A.getValue(6), 1e-10);
    }

    @Test
    public void canSetValueByIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        A.setValue(0.5, 1, 0, 1);
        assertEquals(0.5, A.getValue(5), 1e-10);
    }

    @Test(expected = Exception.class)
    public void doesThrowOnSetByInvalidIndex() {
        DoubleTensor A = DoubleTensor.arange(0, 8).reshape(2, 2, 2);
        A.setValue(0.5, 1, 0);
    }

    @Test
    public void youCanDoTensorYLogXEvenWhenBothAreZero() {
        DoubleTensor zero = DoubleTensor.scalar(0.);
        DoubleTensor zeroTensor = DoubleTensor.create(0., 0., 0.);
        assertThat(zero.log().times(zeroTensor), hasValue(Double.NaN, Double.NaN, Double.NaN));
        assertThat(zero.safeLogTimes(zeroTensor), hasValue(0., 0., 0.));
    }

    @Test
    public void canArgFindMaxOfScalar() {
        DoubleTensor tensor = DoubleTensor.scalar(1).reshape(1, 1);

        assertEquals(0, tensor.argMax());
        assertThat(tensor.argMax(0), valuesAndShapesMatch(IntegerTensor.vector(0)));
        assertThat(tensor.argMax(1), valuesAndShapesMatch(IntegerTensor.vector(0)));
    }

    @Test
    public void comparesScalarWithDoubleTensor() {
        DoubleTensor value = DoubleTensor.scalar(1.);
        DoubleTensor differentValue = DoubleTensor.create(1., 2., 3.);
        BooleanTensor result = value.elementwiseEquals(differentValue);
        assertThat(result, hasValue(true, false, false));
    }

    @Test
    public void doesKeepRankOnGTEq() {
        DoubleTensor value = DoubleTensor.create(new double[]{1}, 1, 1, 1);
        assertEquals(3, value.greaterThanOrEqual(2.0).getRank());
    }

    @Test
    public void doesKeepRankOnGT() {
        DoubleTensor value = DoubleTensor.create(new double[]{1}, 1, 1, 1);
        assertEquals(3, value.greaterThan(2.0).getRank());
    }

    @Test
    public void doesKeepRankOnLT() {
        DoubleTensor value = DoubleTensor.create(new double[]{1}, 1, 1, 1);
        assertEquals(3, value.lessThan(2.0).getRank());
    }

    @Test
    public void doesKeepRankOnLTEq() {
        DoubleTensor value = DoubleTensor.create(new double[]{1}, 1, 1, 1);
        assertEquals(3, value.lessThanOrEqual(2.0).getRank());
    }

    @Test
    public void doesMatrixMultiplyWhen1x1() {
        DoubleTensor lengthOne = DoubleTensor.scalar(2).reshape(1, 1);
        DoubleTensor matrix = DoubleTensor.create(3, 4).reshape(1, 2);
        DoubleTensor result = lengthOne.matrixMultiply(matrix);
        DoubleTensor expected = DoubleTensor.create(6, 8).reshape(1, 2);
        assertEquals(expected, result);
    }

    @Test
    public void doesThrowOnMatrixMultiplyWhenRank0() {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Matrix multiply must be used on matrices");

        DoubleTensor lengthOne = DoubleTensor.scalar(2);
        DoubleTensor matrix = DoubleTensor.create(1, 2, 3, 4).reshape(2, 2);
        lengthOne.matrixMultiply(matrix);
    }

    @Test
    public void doesTensorMultiplyWithScalar() {
        DoubleTensor lengthOne = DoubleTensor.scalar(2).reshape(1);
        DoubleTensor matrix = DoubleTensor.arange(0, 4).reshape(2, 1, 2);
        DoubleTensor result = lengthOne.tensorMultiply(matrix, new int[]{0}, new int[]{1});
        DoubleTensor expected = DoubleTensor.create(0, 2, 4, 6).reshape(2, 2);
        assertEquals(expected, result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidLeftDimsTensorMultiply() {
        DoubleTensor lengthOne = DoubleTensor.scalar(2);
        DoubleTensor matrix = DoubleTensor.arange(0, 4).reshape(2, 1, 2);
        lengthOne.tensorMultiply(matrix, new int[]{0}, new int[]{1});
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesThrowOnInvalidRightDimsTensorMultiply() {
        DoubleTensor lengthOne = DoubleTensor.scalar(2).reshape(1);
        DoubleTensor matrix = DoubleTensor.arange(0, 4).reshape(2, 1, 2);
        lengthOne.tensorMultiply(matrix, new int[]{0}, new int[]{3});
    }

    @Test
    public void canCumSumOnRank3() {
        DoubleTensor a = DoubleTensor.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 2, 2);
        DoubleTensor expected0 = DoubleTensor.create(1, 2, 3, 4, 6, 8, 10, 12).reshape(2, 2, 2);
        DoubleTensor expected1 = DoubleTensor.create(1, 2, 4, 6, 5, 6, 12, 14).reshape(2, 2, 2);
        DoubleTensor expected2 = DoubleTensor.create(1, 3, 3, 7, 5, 11, 7, 15).reshape(2, 2, 2);

        assertThat(a.cumSum(0), valuesAndShapesMatch(expected0));
        assertThat(a.cumSum(1), valuesAndShapesMatch(expected1));
        assertThat(a.cumSum(2), valuesAndShapesMatch(expected2));
        assertThat(a.cumSum(-1), valuesAndShapesMatch(expected2));
        assertThat(a.cumSum(-2), valuesAndShapesMatch(expected1));
        assertThat(a.cumSum(-3), valuesAndShapesMatch(expected0));
    }

    @Test
    public void canCumSumOnMatrix() {
        DoubleTensor a = DoubleTensor.create(1, 2, 3, 4, 5, 6, 7, 8, 9).reshape(3, 3);
        DoubleTensor expected0 = DoubleTensor.create(1, 2, 3, 5, 7, 9, 12, 15, 18).reshape(3, 3);
        DoubleTensor expected1 = DoubleTensor.create(1, 3, 6, 4, 9, 15, 7, 15, 24).reshape(3, 3);

        assertThat(a.cumSum(0), valuesAndShapesMatch(expected0));
        assertThat(a.cumSum(1), valuesAndShapesMatch(expected1));
        assertThat(a.cumSum(-1), valuesAndShapesMatch(expected1));
        assertThat(a.cumSum(-2), valuesAndShapesMatch(expected0));
    }

    @Test
    public void canCumSumOnVector() {
        DoubleTensor a = DoubleTensor.create(1, 2, 3);
        DoubleTensor expected0 = DoubleTensor.create(1, 3, 6);

        assertThat(a.cumSum(0), valuesAndShapesMatch(expected0));
        assertThat(a.cumSum(-1), valuesAndShapesMatch(expected0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfCumSumOnInvalidDimension() {
        DoubleTensor.scalar(2).cumSum(0);
    }

    @Test
    public void canCumProdOnMatrix() {
        DoubleTensor a = DoubleTensor.create(1, 2, 3, 4, 5, 6, 7, 8, 9).reshape(3, 3);
        DoubleTensor expected0 = DoubleTensor.create(1, 2, 3, 4, 10, 18, 28, 80, 162).reshape(3, 3);
        DoubleTensor expected1 = DoubleTensor.create(1, 2, 6, 4, 20, 120, 7, 56, 504).reshape(3, 3);

        assertThat(a.cumProd(0), valuesAndShapesMatch(expected0));
        assertThat(a.cumProd(1), valuesAndShapesMatch(expected1));
        assertThat(a.cumProd(-1), valuesAndShapesMatch(expected1));
        assertThat(a.cumProd(-2), valuesAndShapesMatch(expected0));
    }

    @Test
    public void canCumProdOnVector() {
        DoubleTensor a = DoubleTensor.create(1, 2, 3);
        DoubleTensor expected0 = DoubleTensor.create(1, 2, 6);

        assertThat(a.cumProd(0), valuesAndShapesMatch(expected0));
        assertThat(a.cumProd(-1), valuesAndShapesMatch(expected0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfCumProdOnInvalidDimension() {
        DoubleTensor.scalar(2).cumProd(0);
    }

    @Test
    public void canBroadcastToShape() {
        DoubleTensor a = DoubleTensor.create(
            1, 2, 3
        ).reshape(3);

        DoubleTensor expectedByRow = DoubleTensor.create(
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expectedByRow));

        DoubleTensor expectedByColumn = DoubleTensor.create(
            1, 1, 1,
            2, 2, 2,
            3, 3, 3
        ).reshape(3, 3);

        Assert.assertThat(a.reshape(3, 1).broadcast(3, 3), valuesAndShapesMatch(expectedByColumn));
    }

    @Test
    public void canBroadcastScalarToShape() {
        DoubleTensor a = DoubleTensor.scalar(2);

        DoubleTensor expected = DoubleTensor.create(
            2, 2, 2,
            2, 2, 2,
            2, 2, 2
        ).reshape(3, 3);

        Assert.assertThat(a.broadcast(3, 3), valuesAndShapesMatch(expected));
    }

    @Test
    public void canStartStopStepSlice() {
        /**
         * a = np.arange(10)
         * b = a[2:7:2]
         * print b
         * [2  4  6]
         */

        DoubleTensor a = DoubleTensor.arange(10);

        Slicer slicer = Slicer.builder()
            .slice(2, 7, 2)
            .build();

        DoubleTensor b = a.slice(slicer);

        assertThat(b, valuesAndShapesMatch(DoubleTensor.create(2, 4, 6)));
    }

}
