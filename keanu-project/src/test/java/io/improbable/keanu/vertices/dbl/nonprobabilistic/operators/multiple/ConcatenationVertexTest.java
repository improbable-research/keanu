package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsWithRespectTo;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertEquals;

public class ConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3}, 1, 3));

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(DoubleTensor.create(new double[]{4, 5, 6}, 1, 3));

        UniformVertex c = new UniformVertex(0.0, 1.0);
        c.setValue(DoubleTensor.create(new double[]{7, 8, 9}, 1, 3));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);
        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new long[]{2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new long[]{1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{9}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatZero.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        DoubleVertex b = new ConstantDoubleVertex(new double[]{4.0});

        ConcatenationVertex concat = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        DoubleVertex a = new ConstantDoubleVertex(new double[]{1.0});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{2, 3, 4});

        ConcatenationVertex concat = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3}, 1, 3));

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(DoubleTensor.create(new double[]{4, 5, 6, 7, 8, 9}, 1, 6));

        new ConcatenationVertex(0, a, a1);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new long[]{4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 10, 15, 20, 25}, concatZero.getValue().asFlatDoubleArray(), 0.001);

        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new long[]{2, 4}, concatOne.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 10, 15, 3, 4, 20, 25}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatHighDimensionalShapes() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, 2, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80},
            concatZero.getValue().asFlatDoubleArray(),
            0.001
        );
        Assert.assertArrayEquals(new long[]{4, 2, 2}, concatZero.getShape());

        ConcatenationVertex concatThree = new ConcatenationVertex(2, a, b);
        Assert.assertArrayEquals(
            new double[]{1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80},
            concatThree.getValue().asFlatDoubleArray(),
            0.001
        );
        Assert.assertArrayEquals(new long[]{2, 2, 4}, concatThree.getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiff() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        MultiplicationVertex c = a.times(b);
        AdditionVertex d = a.plus(b);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);

        DoubleTensor dConcatWrtA = Differentiator.forwardModeAutoDiff(a, concat).of(concat);
        DoubleTensor dConcatWrtB = Differentiator.forwardModeAutoDiff(b, concat).of(concat);

        PartialsWithRespectTo wrtAForward = Differentiator.forwardModeAutoDiff(a, c, d);
        PartialsWithRespectTo wrtBForward = Differentiator.forwardModeAutoDiff(b, c, d);

        DoubleTensor dCWrtA = wrtAForward.of(c);
        DoubleTensor dDWrtA = wrtAForward.of(d);
        DoubleTensor dCWrtB = wrtBForward.of(c);
        DoubleTensor dDWrtB = wrtBForward.of(d);

        Assert.assertArrayEquals(
            DoubleTensor.concat(0, dCWrtA, dDWrtA).asFlatDoubleArray(),
            dConcatWrtA.asFlatDoubleArray(),
            0.0001
        );

        Assert.assertArrayEquals(
            DoubleTensor.concat(0, dCWrtB, dDWrtB).asFlatDoubleArray(),
            dConcatWrtB.asFlatDoubleArray(),
            0.0001
        );

        PartialsOf concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, a, b);

        assertEquals(dConcatWrtA, concatPartialReverse.withRespectTo(a));
        assertEquals(dConcatWrtB, concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateHighRankAutoDiff() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 12).reshape(2, 2, 3));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex c = a.times(ConstantVertex.of(DoubleTensor.linspace(0, 1, 12).reshape(2, 2, 3)));
        DoubleVertex d = b.plus(ConstantVertex.of(DoubleTensor.linspace(1, 2, 8).reshape(2, 2, 2)));

        DoubleVertex concat = new ConcatenationVertex(2, c, d);
        SumVertex sum = concat.sum(1);

        finiteDifferenceMatchesForwardAndReverseModeGradient(Arrays.asList(a, b), sum, 10.0, 1e-10);
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardNoSharedParentsDimensionOne() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        UniformVertex c = new UniformVertex(0, 10);
        c.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        UniformVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = c.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, e, f);

        PartialsOf reverse = Differentiator.reverseModeAutoDiff(concat, a, b, c, d);

        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, Differentiator.forwardModeAutoDiff(a, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, Differentiator.forwardModeAutoDiff(b, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, Differentiator.forwardModeAutoDiff(c, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, Differentiator.forwardModeAutoDiff(d, concat).of(concat).getShape());

        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, reverse.withRespectTo(c).getShape());
        Assert.assertArrayEquals(new long[]{2, 4, 2, 2}, reverse.withRespectTo(d).getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardSharedParents() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 1, 1));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        UniformVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = b.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(0, e, f);

        PartialsOf reverse = Differentiator.reverseModeAutoDiff(concat, a, b, d);

        Assert.assertArrayEquals(new long[]{4, 2, 1, 1}, Differentiator.forwardModeAutoDiff(a, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{4, 2, 2, 2}, Differentiator.forwardModeAutoDiff(b, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{4, 2, 2, 2}, Differentiator.forwardModeAutoDiff(d, concat).of(concat).getShape());

        Assert.assertArrayEquals(new long[]{4, 2, 1, 1}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new long[]{4, 2, 2, 2}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new long[]{4, 2, 2, 2}, reverse.withRespectTo(d).getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardSharedParentsAndDifferentSize() {
        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 3));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 3));

        UniformVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 3, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = b.matrixMultiply(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, e, f);

        PartialsOf reverse = Differentiator.reverseModeAutoDiff(concat, a, b, d);

        Assert.assertArrayEquals(new long[]{2, 5, 2, 3}, Differentiator.forwardModeAutoDiff(a, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{2, 5, 2, 3}, Differentiator.forwardModeAutoDiff(b, concat).of(concat).getShape());
        Assert.assertArrayEquals(new long[]{2, 5, 3, 2}, Differentiator.forwardModeAutoDiff(d, concat).of(concat).getShape());

        Assert.assertArrayEquals(new long[]{2, 5, 2, 3}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new long[]{2, 5, 2, 3}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new long[]{2, 5, 3, 2}, reverse.withRespectTo(d).getShape());
    }

    @Test
    public void canCalculateValueOfConcatenated() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = a.plus(b);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        DoubleTensor concatResult = concat.eval();

        Assert.assertArrayEquals(
            new double[]{50, 90, 140, 200, 15, 21, 27, 33},
            concatResult.asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionZero() {
        UniformVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        MatrixMultiplicationVertex c = sharedMatrix.matrixMultiply(a);
        MatrixMultiplicationVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, c).of(c);
        DoubleTensor dDdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, d).of(d);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);

        DoubleTensor dConcatWrtAForward = Differentiator.forwardModeAutoDiff(a, concat).of(concat);
        DoubleTensor dConcatWrtSharedMatrixForward = Differentiator.forwardModeAutoDiff(sharedMatrix, concat).of(concat);
        DoubleTensor dConcatWrtBForward = Differentiator.forwardModeAutoDiff(b, concat).of(concat);

        PartialsOf concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b);

        Assert.assertArrayEquals(
            DoubleTensor.concat(0, dCdshared, dDdshared).asFlatDoubleArray(),
            dConcatWrtSharedMatrixForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtSharedMatrixForward, concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = Differentiator.forwardModeAutoDiff(a, c).of(c);
        Assert.assertArrayEquals(
            DoubleTensor.concat(0, cwrtA, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            dConcatWrtAForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtAForward, concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = Differentiator.forwardModeAutoDiff(b, d).of(d);
        Assert.assertArrayEquals(
            DoubleTensor.concat(0, DoubleTensor.zeros(dwrtB.getShape()), dwrtB).asFlatDoubleArray(),
            dConcatWrtBForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtBForward, concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionOne() {
        UniformVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        MatrixMultiplicationVertex c = sharedMatrix.matrixMultiply(a);
        MatrixMultiplicationVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, c).of(c);
        DoubleTensor dDdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, d).of(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, c, d);

        DoubleTensor dConcatWrtSharedMatrixForward = Differentiator.forwardModeAutoDiff(sharedMatrix, concat).of(concat);
        DoubleTensor dConcatWrtAForward = Differentiator.forwardModeAutoDiff(a, concat).of(concat);
        DoubleTensor dConcatWrtBForward = Differentiator.forwardModeAutoDiff(b, concat).of(concat);

        PartialsOf concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b);

        Assert.assertArrayEquals(
            DoubleTensor.concat(1, dCdshared, dDdshared).asFlatDoubleArray(),
            dConcatWrtSharedMatrixForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtSharedMatrixForward, concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = Differentiator.forwardModeAutoDiff(a, c).of(c);
        Assert.assertArrayEquals(
            DoubleTensor.concat(1, cwrtA, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            dConcatWrtAForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtAForward, concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = Differentiator.forwardModeAutoDiff(b, d).of(d);
        Assert.assertArrayEquals(
            DoubleTensor.concat(1, DoubleTensor.zeros(dwrtB.getShape()), dwrtB).asFlatDoubleArray(),
            dConcatWrtBForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtBForward, concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateAutoDiffOfManyMatricesAlongDimensionZero() {
        UniformVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        UniformVertex f = new UniformVertex(0, 10);
        f.setValue(DoubleTensor.create(new double[]{90, 91, 92, 93}, 2, 2));

        MatrixMultiplicationVertex c = sharedMatrix.matrixMultiply(a);
        MatrixMultiplicationVertex d = sharedMatrix.matrixMultiply(b);
        MatrixMultiplicationVertex e = sharedMatrix.matrixMultiply(f);

        DoubleTensor dCdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, c).of(c);
        DoubleTensor dDdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, d).of(d);
        DoubleTensor dEdshared = Differentiator.forwardModeAutoDiff(sharedMatrix, e).of(e);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d, e);
        PartialsOf concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b, f);

        DoubleTensor dConcatWrtSharedMatrixForward = Differentiator.forwardModeAutoDiff(sharedMatrix, concat).of(concat);
        DoubleTensor dConcatWrtAForward = Differentiator.forwardModeAutoDiff(a, concat).of(concat);
        DoubleTensor dConcatWrtBForward = Differentiator.forwardModeAutoDiff(b, concat).of(concat);
        DoubleTensor dConcatWrtFForward = Differentiator.forwardModeAutoDiff(f, concat).of(concat);

        Assert.assertArrayEquals(
            DoubleTensor.concat(0, dCdshared, dDdshared, dEdshared).asFlatDoubleArray(),
            dConcatWrtSharedMatrixForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtSharedMatrixForward, concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = Differentiator.forwardModeAutoDiff(a, c).of(c);
        Assert.assertArrayEquals(
            DoubleTensor.concat(0, cwrtA, DoubleTensor.zeros(cwrtA.getShape()), DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            dConcatWrtAForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtAForward, concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = Differentiator.forwardModeAutoDiff(b, d).of(d);
        Assert.assertArrayEquals(
            DoubleTensor.concat(0, DoubleTensor.zeros(dwrtB.getShape()), dwrtB, DoubleTensor.zeros(dwrtB.getShape())).asFlatDoubleArray(),
            dConcatWrtBForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtBForward, concatPartialReverse.withRespectTo(b));

        DoubleTensor ewrtC = Differentiator.forwardModeAutoDiff(f, e).of(e);
        Assert.assertArrayEquals(
            DoubleTensor.concat(0, DoubleTensor.zeros(ewrtC.getShape()), DoubleTensor.zeros(ewrtC.getShape()), ewrtC).asFlatDoubleArray(),
            dConcatWrtFForward.asFlatDoubleArray(),
            0.0001
        );

        assertEquals(dConcatWrtFForward, concatPartialReverse.withRespectTo(f));
    }

    @Test
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        UniformVertex inputB = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        UniformVertex inputC = new UniformVertex(new long[]{2, 2, 2}, -10.0, 10.0);
        ConcatenationVertex outputVertex = new ConcatenationVertex(0, inputA, inputB, inputC);
        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA, inputB, inputC), outputVertex, 10.0, 1e-10);
    }

}
