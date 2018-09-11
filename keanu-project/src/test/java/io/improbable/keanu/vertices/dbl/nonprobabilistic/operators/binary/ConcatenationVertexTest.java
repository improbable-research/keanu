package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static org.junit.Assert.assertEquals;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ConcatenationVertexTest {

    @Test
    public void canConcatVectorsOfSameSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{4, 5, 6});

        UniformVertex c = new UniformVertex(0.0, 1.0);
        c.setValue(new double[]{7, 8, 9});

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);
        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b, c);

        Assert.assertArrayEquals(new int[]{2, 3}, concatZero.getShape());
        Assert.assertArrayEquals(new int[]{1, 9}, concatOne.getShape());

        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, concatZero.getValue().asFlatDoubleArray(), 0.001);
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatOne.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorsOfDifferentSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{4, 5, 6, 7, 8, 9});

        ConcatenationVertex concatZero = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 9}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, concatZero.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatScalarToVector() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        DoubleVertex b = new ConstantDoubleVertex(4.0);

        ConcatenationVertex concat = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test
    public void canConcatVectorToScalar() {
        DoubleVertex a = new ConstantDoubleVertex(1.0);

        UniformVertex b = new UniformVertex(0.0, 1.0);
        b.setValue(new double[]{2, 3, 4});

        ConcatenationVertex concat = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{1, 4}, concat.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4}, concat.getValue().asFlatDoubleArray(), 0.001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void errorThrownOnConcatOfWrongSize() {
        UniformVertex a = new UniformVertex(0.0, 1.0);
        a.setValue(new double[]{1, 2, 3});

        UniformVertex a1 = new UniformVertex(0.0, 1.0);
        a1.setValue(new double[]{4, 5, 6, 7, 8, 9});

        new ConcatenationVertex(0, a, a1);
    }

    @Test
    public void canConcatMatricesOfSameSize() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        ConcatenationVertex concatZero = new ConcatenationVertex(0, a, b);

        Assert.assertArrayEquals(new int[]{4, 2}, concatZero.getShape());
        Assert.assertArrayEquals(new double[]{1, 2, 3, 4, 10, 15, 20, 25}, concatZero.getValue().asFlatDoubleArray(), 0.001);

        ConcatenationVertex concatOne = new ConcatenationVertex(1, a, b);

        Assert.assertArrayEquals(new int[]{2, 4}, concatOne.getShape());
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
        Assert.assertArrayEquals(new int[]{4, 2, 2}, concatZero.getShape());

        ConcatenationVertex concatThree = new ConcatenationVertex(2, a, b);
        Assert.assertArrayEquals(
            new double[]{1, 2, 10, 20, 3, 4, 30, 40, 5, 6, 50, 60, 7, 8, 70, 80},
            concatThree.getValue().asFlatDoubleArray(),
            0.001
        );
        Assert.assertArrayEquals(new int[]{2, 2, 4}, concatThree.getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiff() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = a.plus(b);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        PartialDerivatives concatPartialForward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, a, b);

        PartialDerivatives cPartial = c.getDualNumber().getPartialDerivatives();
        PartialDerivatives dPartial = d.getDualNumber().getPartialDerivatives();

        Assert.assertArrayEquals(
            cPartial.withRespectTo(a).concat(0, dPartial.withRespectTo(a)).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        Assert.assertArrayEquals(
            cPartial.withRespectTo(b).concat(0, dPartial.withRespectTo(b)).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(a), concatPartialReverse.withRespectTo(a));
        assertEquals(concatPartialForward.withRespectTo(b), concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardNoSharedParentsDimensionOne() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = new UniformVertex(0, 10);
        c.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = c.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, e, f);

        PartialDerivatives forward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives reverse = Differentiator.reverseModeAutoDiff(concat, a, b, c, d);

        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(c).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, forward.withRespectTo(d).getShape());

        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, reverse.withRespectTo(c).getShape());
        Assert.assertArrayEquals(new int[]{2, 4, 2, 2}, reverse.withRespectTo(d).getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardSharedParents() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5}, 1, 1));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = b.plus(d);

        ConcatenationVertex concat = new ConcatenationVertex(0, e, f);

        PartialDerivatives forward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives reverse = Differentiator.reverseModeAutoDiff(concat, a, b, d);

        Assert.assertArrayEquals(new int[]{4, 2, 1, 1}, forward.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, forward.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, forward.withRespectTo(d).getShape());

        Assert.assertArrayEquals(new int[]{4, 2, 1, 1}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{4, 2, 2, 2}, reverse.withRespectTo(d).getShape());
    }

    @Test
    public void canConcatenateSimpleAutoDiffForwardSharedParentsAndDifferentSize() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 3));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 3));

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 3, 2));

        DoubleVertex e = a.times(b);
        DoubleVertex f = b.matrixMultiply(d);

        ConcatenationVertex concat = new ConcatenationVertex(1, e, f);

        PartialDerivatives forward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives reverse = Differentiator.reverseModeAutoDiff(concat, a, b, d);

        Assert.assertArrayEquals(new int[]{2, 5, 2, 3}, forward.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{2, 5, 2, 3}, forward.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{2, 5, 3, 2}, forward.withRespectTo(d).getShape());

        Assert.assertArrayEquals(new int[]{2, 5, 2, 3}, reverse.withRespectTo(a).getShape());
        Assert.assertArrayEquals(new int[]{2, 5, 2, 3}, reverse.withRespectTo(b).getShape());
        Assert.assertArrayEquals(new int[]{2, 5, 3, 2}, reverse.withRespectTo(d).getShape());
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
        DoubleTensor dualNumberValue = concat.eval();

        Assert.assertArrayEquals(
            new double[]{50, 90, 140, 200, 15, 21, 27, 33},
            dualNumberValue.asFlatDoubleArray(),
            0.0001
        );
    }

    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionZero() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d);
        PartialDerivatives concatPartialForward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b);

        Assert.assertArrayEquals(
            dCdshared.concat(0, dDdshared).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(sharedMatrix), concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(0, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(a), concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(0, dwrtB).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(b), concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateAutoDiffMatricesAlongDimensionOne() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(1, c, d);
        PartialDerivatives concatPartialForward = concat.getDualNumber().getPartialDerivatives();

        PartialDerivatives concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b);

        Assert.assertArrayEquals(
            dCdshared.concat(1, dDdshared).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(sharedMatrix), concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(1, DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(a), concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(1, dwrtB).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(b), concatPartialReverse.withRespectTo(b));
    }

    @Test
    public void canConcatenateAutoDiffOfManyMatricesAlongDimensionZero() {
        DoubleVertex sharedMatrix = new UniformVertex(0, 10);
        sharedMatrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{10, 15, 20, 25}, 2, 2));

        DoubleVertex f = new UniformVertex(0, 10);
        f.setValue(DoubleTensor.create(new double[]{90, 91, 92, 93}, 2, 2));

        DoubleVertex c = sharedMatrix.matrixMultiply(a);
        DoubleVertex d = sharedMatrix.matrixMultiply(b);
        DoubleVertex e = sharedMatrix.matrixMultiply(f);

        DoubleTensor dCdshared = c.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dDdshared = d.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);
        DoubleTensor dEdshared = e.getDualNumber().getPartialDerivatives().withRespectTo(sharedMatrix);

        ConcatenationVertex concat = new ConcatenationVertex(0, c, d, e);
        PartialDerivatives concatPartialForward = concat.getDualNumber().getPartialDerivatives();
        PartialDerivatives concatPartialReverse = Differentiator.reverseModeAutoDiff(concat, sharedMatrix, a, b, f);

        Assert.assertArrayEquals(
            dCdshared.concat(0, dDdshared, dEdshared).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(sharedMatrix).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(sharedMatrix), concatPartialReverse.withRespectTo(sharedMatrix));

        DoubleTensor cwrtA = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        Assert.assertArrayEquals(
            cwrtA.concat(0, DoubleTensor.zeros(cwrtA.getShape()), DoubleTensor.zeros(cwrtA.getShape())).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(a).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(a), concatPartialReverse.withRespectTo(a));

        DoubleTensor dwrtB = d.getDualNumber().getPartialDerivatives().withRespectTo(b);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(dwrtB.getShape()).concat(0, dwrtB, DoubleTensor.zeros(dwrtB.getShape())).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(b).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(b), concatPartialReverse.withRespectTo(b));

        DoubleTensor ewrtC = e.getDualNumber().getPartialDerivatives().withRespectTo(f);
        Assert.assertArrayEquals(
            DoubleTensor.zeros(ewrtC.getShape()).concat(0, DoubleTensor.zeros(ewrtC.getShape()), ewrtC).asFlatDoubleArray(),
            concatPartialForward.withRespectTo(f).asFlatDoubleArray(),
            0.0001
        );

        assertEquals(concatPartialForward.withRespectTo(f), concatPartialReverse.withRespectTo(f));
    }

}
